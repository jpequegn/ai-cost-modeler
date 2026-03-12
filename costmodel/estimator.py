"""Cost estimator: compute expected cost from an Architecture description."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path

from costmodel.models import Architecture, ModelCall, Stage, get_architecture
from costmodel.pricing import (
    TASK_INPUT_TOKENS,
    TASK_OUTPUT_TOKENS,
    cost_for_call,
    resolve_model,
)


@dataclass
class StageCostBreakdown:
    """Cost breakdown for a single stage."""

    stage_name: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    calls_count: int  # total effective calls (repeats * condition)


@dataclass
class CostEstimate:
    """Estimated cost for one run of an architecture."""

    architecture_name: str
    per_run_usd: float
    per_1000_runs_usd: float
    per_stage: list[StageCostBreakdown] = field(default_factory=list)
    confidence: str = "medium"  # "high" | "medium" | "low"
    notes: list[str] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.per_stage)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.per_stage)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


def _estimate_stage(stage: Stage) -> StageCostBreakdown:
    """Compute cost for one stage (one execution, honouring condition probability)."""
    stage_cost = 0.0
    total_input = 0
    total_output = 0
    total_cached = 0
    total_calls = 0

    for call in stage.calls:
        effective_repeats = call.repeats
        call_cost = cost_for_call(
            model=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cached_input_tokens=call.cached_input_tokens,
            cache_write_tokens=call.cache_write_tokens,
        )
        stage_cost += call_cost * effective_repeats
        total_input += call.input_tokens * effective_repeats
        total_output += call.output_tokens * effective_repeats
        total_cached += call.cached_input_tokens * effective_repeats
        total_calls += effective_repeats

    # Apply stage-level condition (probability it runs)
    stage_cost *= stage.condition
    total_input = int(total_input * stage.condition)
    total_output = int(total_output * stage.condition)
    total_cached = int(total_cached * stage.condition)

    return StageCostBreakdown(
        stage_name=stage.name,
        cost_usd=stage_cost,
        input_tokens=total_input,
        output_tokens=total_output,
        cached_input_tokens=total_cached,
        calls_count=total_calls,
    )


def estimate(arch: Architecture) -> CostEstimate:
    """Compute cost estimate for one run of an architecture."""
    stage_breakdowns: list[StageCostBreakdown] = []
    for stage in arch.stages:
        stage_breakdowns.append(_estimate_stage(stage))

    per_run = sum(s.cost_usd for s in stage_breakdowns)
    per_1000 = per_run * 1000

    # Simple confidence heuristic
    has_condition = any(s.condition < 1.0 for s in arch.stages)
    has_cache = any(
        c.cached_input_tokens > 0
        for s in arch.stages
        for c in s.calls
    )
    if has_condition and has_cache:
        confidence = "low"
    elif has_condition or has_cache:
        confidence = "medium"
    else:
        confidence = "high"

    notes: list[str] = []
    if has_condition:
        notes.append("Some stages have conditional execution (probability < 1.0)")
    if has_cache:
        notes.append("Cache usage assumed — actual cache hit rate may differ")

    return CostEstimate(
        architecture_name=arch.name,
        per_run_usd=per_run,
        per_1000_runs_usd=per_1000,
        per_stage=stage_breakdowns,
        confidence=confidence,
        notes=notes,
    )


def estimate_from_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> CostEstimate:
    """Estimate cost for a simple single-call architecture from raw token counts."""
    from costmodel.models import Architecture, Stage, ModelCall

    model_canonical = resolve_model(model)
    arch = Architecture(
        name=f"single-call-{model_canonical}",
        stages=[
            Stage(
                name="main",
                calls=[
                    ModelCall(
                        model=model_canonical,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                    )
                ],
            )
        ],
    )
    return estimate(arch)


def estimate_from_task_type(
    model: str,
    task_type: str,
) -> CostEstimate:
    """Estimate cost using heuristic token counts for a known task type."""
    from costmodel.models import Architecture, Stage, ModelCall

    model_canonical = resolve_model(model)
    input_tokens = TASK_INPUT_TOKENS.get(task_type, 3000)
    output_tokens = TASK_OUTPUT_TOKENS.get(task_type, 400)

    arch = Architecture(
        name=f"{task_type}-{model_canonical}",
        stages=[
            Stage(
                name=task_type,
                calls=[
                    ModelCall(
                        model=model_canonical,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                ],
            )
        ],
    )
    est = estimate(arch)
    est.notes.append(
        f"Token counts from task-type heuristic ({task_type}): "
        f"{input_tokens} in / {output_tokens} out"
    )
    return est


def load_architecture_from_yaml(path: str | Path) -> Architecture:
    """Load an Architecture from a YAML file.

    Expected YAML format::

        name: my-arch
        description: Optional description
        stages:
          - name: planning
            calls:
              - model: claude-opus-4-5
                input_tokens: 2000
                output_tokens: 500
          - name: review
            parallel: true
            calls:
              - model: claude-sonnet-4-5
                input_tokens: 8000
                output_tokens: 1000
                repeats: 3
    """
    data = yaml.safe_load(Path(path).read_text())
    stages: list[Stage] = []
    for s in data.get("stages", []):
        calls: list[ModelCall] = []
        for c in s.get("calls", []):
            calls.append(
                ModelCall(
                    model=c["model"],
                    input_tokens=int(c.get("input_tokens", 0)),
                    output_tokens=int(c.get("output_tokens", 0)),
                    cached_input_tokens=int(c.get("cached_input_tokens", 0)),
                    cache_write_tokens=int(c.get("cache_write_tokens", 0)),
                    repeats=int(c.get("repeats", 1)),
                )
            )
        stages.append(
            Stage(
                name=s["name"],
                calls=calls,
                parallel=bool(s.get("parallel", False)),
                condition=float(s.get("condition", 1.0)),
            )
        )
    return Architecture(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        stages=stages,
        tags=data.get("tags", []),
    )
