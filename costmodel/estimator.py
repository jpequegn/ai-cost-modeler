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

# Tokens-per-second approximation for output token latency
_OUTPUT_TOKENS_PER_SEC = 50.0


@dataclass
class CostEstimate:
    """Estimated cost for one run of an architecture.

    Fields
    ------
    architecture_name : str
    per_run_usd       : float  — total USD cost for one run
    per_1000_runs_usd : float  — per_run_usd × 1 000
    per_stage         : dict[str, float]
        Stage name → expected USD cost (condition probability applied).
    token_breakdown   : dict[str, dict]
        Stage name → ``{"input": int, "output": int, "cache": int}``
        token counts (condition probability applied to counts).
    latency_seconds   : float
        Sequential sum of stage latencies; parallel stages contribute
        ``max(call latencies)`` rather than their sum.
        Per-call latency = ``output_tokens / 50`` (tokens/sec approximation).
    confidence        : str — ``"high"`` | ``"medium"`` | ``"low"``
        ``"high"`` when every token count was supplied explicitly (no
        heuristics used); ``"low"`` if any heuristic estimate was applied;
        ``"medium"`` otherwise (conditional stages, cache assumptions, etc.).
    notes             : list[str]  — optional human-readable annotations
    """

    architecture_name: str
    per_run_usd: float
    per_1000_runs_usd: float
    per_stage: dict[str, float] = field(default_factory=dict)
    token_breakdown: dict[str, dict] = field(default_factory=dict)
    latency_seconds: float = 0.0
    confidence: str = "high"  # "high" | "medium" | "low"
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties (derived from per_stage / token_breakdown)
    # ------------------------------------------------------------------

    @property
    def total_input_tokens(self) -> int:
        return sum(v.get("input", 0) for v in self.token_breakdown.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(v.get("output", 0) for v in self.token_breakdown.values())

    @property
    def total_cached_tokens(self) -> int:
        return sum(v.get("cache", 0) for v in self.token_breakdown.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens + self.total_cached_tokens


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stage_cost_and_tokens(
    stage: Stage,
) -> tuple[float, dict]:
    """Return (cost_usd, token_dict) for one stage execution.

    ``cost_usd`` and token counts both have the condition probability applied.
    ``token_dict`` keys: ``input``, ``output``, ``cache``.
    """
    stage_cost = 0.0
    total_input = 0
    total_output = 0
    total_cache = 0

    for call in stage.calls:
        call_cost = cost_for_call(
            model=call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cached_input_tokens=call.cached_input_tokens,
            cache_write_tokens=call.cache_write_tokens,
        )
        stage_cost += call_cost * call.repeats
        total_input += call.input_tokens * call.repeats
        total_output += call.output_tokens * call.repeats
        total_cache += (call.cached_input_tokens + call.cache_write_tokens) * call.repeats

    # Apply stage-level condition probability
    stage_cost *= stage.condition
    total_input = int(total_input * stage.condition)
    total_output = int(total_output * stage.condition)
    total_cache = int(total_cache * stage.condition)

    return stage_cost, {"input": total_input, "output": total_output, "cache": total_cache}


def _stage_latency(stage: Stage) -> float:
    """Estimate latency in seconds for one stage.

    *Parallel* stages: all calls (including each repeat) run concurrently,
    so the stage latency is the *maximum* single-call latency across all
    calls and repeats.
    *Sequential* stages: calls (and repeats) run one after another, so the
    stage latency is the *sum* of all call latencies.

    Per-call latency = output_tokens / _OUTPUT_TOKENS_PER_SEC (50 tok/s).
    Condition probability does NOT affect latency — if a stage runs, it
    takes the full time.
    """
    if not stage.calls:
        return 0.0

    # Build a flat list of per-invocation latencies
    invocation_latencies = []
    for call in stage.calls:
        per_invocation = call.output_tokens / _OUTPUT_TOKENS_PER_SEC
        for _ in range(call.repeats):
            invocation_latencies.append(per_invocation)

    if stage.parallel:
        # All invocations run concurrently — wall-clock time = slowest
        return max(invocation_latencies)
    else:
        # Sequential — wall-clock time = sum of all
        return sum(invocation_latencies)


def _compute_confidence(arch: Architecture, used_heuristics: bool) -> str:
    """Determine confidence level.

    * ``"high"``   — all token counts are explicit (no heuristics).
    * ``"low"``    — heuristic token estimates were used.
    * ``"medium"`` — token counts are explicit but some stages are
                     conditional or use cache assumptions.
    """
    if used_heuristics:
        return "low"

    has_condition = any(s.condition < 1.0 for s in arch.stages)
    has_cache = any(
        c.cached_input_tokens > 0 or c.cache_write_tokens > 0
        for s in arch.stages
        for c in s.calls
    )

    if has_condition or has_cache:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate(arch: Architecture, *, _used_heuristics: bool = False) -> "CostEstimate":
    """Compute cost estimate for one run of an architecture.

    Parameters
    ----------
    arch:
        An :class:`~costmodel.models.Architecture` describing the pipeline.
    _used_heuristics:
        Internal flag set by helper functions (``estimate_from_task_type``)
        that build the architecture using heuristic token counts.  When
        ``True`` the confidence is set to ``"low"``.

    Returns
    -------
    CostEstimate
    """
    per_stage: dict[str, float] = {}
    token_breakdown: dict[str, dict] = {}
    total_latency = 0.0
    notes: list[str] = []

    for stage in arch.stages:
        cost, tokens = _stage_cost_and_tokens(stage)
        latency = _stage_latency(stage)

        # Deduplicate stage names by appending an index if needed
        stage_key = stage.name
        if stage_key in per_stage:
            idx = 2
            while f"{stage_key}_{idx}" in per_stage:
                idx += 1
            stage_key = f"{stage_key}_{idx}"

        per_stage[stage_key] = cost
        token_breakdown[stage_key] = tokens

        # Latency is always sequential across stages (sum); within a stage,
        # _stage_latency already handles parallel vs sequential.
        total_latency += latency

    per_run = sum(per_stage.values())
    per_1000 = per_run * 1000

    confidence = _compute_confidence(arch, used_heuristics=_used_heuristics)

    has_condition = any(s.condition < 1.0 for s in arch.stages)
    has_cache = any(
        c.cached_input_tokens > 0 or c.cache_write_tokens > 0
        for s in arch.stages
        for c in s.calls
    )
    if has_condition:
        notes.append("Some stages have conditional execution (probability < 1.0)")
    if has_cache:
        notes.append("Cache usage assumed — actual cache hit rate may differ")
    if _used_heuristics:
        notes.append("Token counts derived from task-type heuristics")

    return CostEstimate(
        architecture_name=arch.name,
        per_run_usd=per_run,
        per_1000_runs_usd=per_1000,
        per_stage=per_stage,
        token_breakdown=token_breakdown,
        latency_seconds=total_latency,
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

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
    est = estimate(arch, _used_heuristics=True)
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
