"""Architecture dataclasses: Pipeline, Stage, ModelCall.

These describe *expected* (estimated) architectures before running them.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelCall:
    """A single LLM API call within a stage."""

    model: str
    input_tokens: int  # estimated uncached input tokens
    output_tokens: int  # estimated output tokens
    cached_input_tokens: int = 0  # tokens expected to hit cache
    cache_write_tokens: int = 0  # tokens written to cache on first call
    repeats: int = 1  # how many times this call happens per stage execution


@dataclass
class Stage:
    """A named processing stage containing one or more model calls."""

    name: str
    calls: list[ModelCall]
    parallel: bool = False  # calls run in parallel (affects latency, not cost)
    condition: float = 1.0  # probability this stage runs (0.0–1.0)


@dataclass
class Architecture:
    """A complete pipeline architecture to estimate costs for."""

    name: str
    stages: list[Stage]
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # YAML serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Architecture":
        """Load an Architecture from a YAML file.

        Expected YAML format::

            name: my-arch
            description: Optional description
            tags:
              - code-review
            stages:
              - name: planning
                calls:
                  - model: claude-opus-4-5
                    input_tokens: 2000
                    output_tokens: 500
              - name: review
                parallel: true
                condition: 0.8
                calls:
                  - model: claude-sonnet-4-5
                    input_tokens: 8000
                    output_tokens: 1000
                    repeats: 3

        Args:
            path: Path to the YAML file.

        Returns:
            An :class:`Architecture` instance.
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
        return cls(
            name=data.get("name", Path(path).stem),
            description=data.get("description", ""),
            stages=stages,
            tags=data.get("tags", []),
        )

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise this Architecture to YAML.

        Args:
            path: If given, write the YAML to this file in addition to
                  returning it as a string.

        Returns:
            YAML string representation of the architecture.
        """
        data: dict = {
            "name": self.name,
            "description": self.description,
        }
        if self.tags:
            data["tags"] = list(self.tags)
        data["stages"] = []
        for stage in self.stages:
            s: dict = {"name": stage.name}
            if stage.parallel:
                s["parallel"] = True
            if stage.condition != 1.0:
                s["condition"] = stage.condition
            s["calls"] = []
            for call in stage.calls:
                c: dict = {
                    "model": call.model,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                }
                if call.cached_input_tokens:
                    c["cached_input_tokens"] = call.cached_input_tokens
                if call.cache_write_tokens:
                    c["cache_write_tokens"] = call.cache_write_tokens
                if call.repeats != 1:
                    c["repeats"] = call.repeats
                s["calls"].append(c)
            data["stages"].append(s)

        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text


# ---------------------------------------------------------------------------
# Built-in named architectures (used when --arch is a short name)
# ---------------------------------------------------------------------------

def _make_builtin_architectures() -> dict[str, Architecture]:
    """Return the library of built-in architecture templates."""
    archs: dict[str, Architecture] = {}

    # ── Single-agent architectures ──────────────────────────────────────────
    archs["single-agent-haiku"] = Architecture(
        name="single-agent-haiku",
        description="Single Haiku agent: one call per task",
        tags=["single-agent", "haiku"],
        stages=[
            Stage(
                name="main",
                calls=[ModelCall("claude-haiku-4-5", input_tokens=3000, output_tokens=400)],
            )
        ],
    )

    archs["single-agent-sonnet"] = Architecture(
        name="single-agent-sonnet",
        description="Single Sonnet agent: one call per task",
        tags=["single-agent", "sonnet"],
        stages=[
            Stage(
                name="main",
                calls=[ModelCall("claude-sonnet-4-5", input_tokens=3000, output_tokens=400)],
            )
        ],
    )

    archs["single-agent-opus"] = Architecture(
        name="single-agent-opus",
        description="Single Opus agent: one call per task",
        tags=["single-agent", "opus"],
        stages=[
            Stage(
                name="main",
                calls=[ModelCall("claude-opus-4-5", input_tokens=3000, output_tokens=400)],
            )
        ],
    )

    # ── Code-review architectures ────────────────────────────────────────────
    archs["code-review-haiku"] = Architecture(
        name="code-review-haiku",
        description="Code review with Haiku",
        tags=["code-review", "haiku"],
        stages=[
            Stage(
                name="review",
                calls=[ModelCall("claude-haiku-4-5", input_tokens=8000, output_tokens=400)],
            )
        ],
    )

    archs["code-review-sonnet"] = Architecture(
        name="code-review-sonnet",
        description="Code review with Sonnet",
        tags=["code-review", "sonnet"],
        stages=[
            Stage(
                name="review",
                calls=[ModelCall("claude-sonnet-4-5", input_tokens=8000, output_tokens=400)],
            )
        ],
    )

    archs["code-review-opus"] = Architecture(
        name="code-review-opus",
        description="Code review with Opus",
        tags=["code-review", "opus"],
        stages=[
            Stage(
                name="review",
                calls=[ModelCall("claude-opus-4-5", input_tokens=8000, output_tokens=400)],
            )
        ],
    )

    # ── 3-agent parallel review ──────────────────────────────────────────────
    archs["3-agent-sonnet"] = Architecture(
        name="3-agent-sonnet",
        description="Planning (Opus) + 3 parallel Sonnet reviewers + synthesis (Opus)",
        tags=["multi-agent", "code-review", "sonnet", "opus"],
        stages=[
            Stage(
                name="planning",
                calls=[ModelCall("claude-opus-4-5", input_tokens=2000, output_tokens=500)],
            ),
            Stage(
                name="review",
                calls=[ModelCall("claude-sonnet-4-5", input_tokens=8000, output_tokens=1000, repeats=3)],
                parallel=True,
            ),
            Stage(
                name="synthesis",
                calls=[ModelCall("claude-opus-4-5", input_tokens=5000, output_tokens=800)],
            ),
        ],
    )

    # ── Anthropic's commercial code review ($20 baseline) ────────────────────
    archs["anthropic-code-review"] = Architecture(
        name="anthropic-code-review",
        description="Anthropic's commercial code review product (~$20/review)",
        tags=["baseline", "commercial"],
        stages=[
            Stage(
                name="commercial",
                calls=[ModelCall("claude-opus-4-5", input_tokens=50000, output_tokens=5000)],
            )
        ],
    )

    return archs


BUILTIN_ARCHITECTURES: dict[str, Architecture] = _make_builtin_architectures()


def get_architecture(name: str) -> Architecture | None:
    """Return a built-in architecture by name, or None if not found."""
    return BUILTIN_ARCHITECTURES.get(name)


def list_architectures() -> list[str]:
    """Return names of all built-in architectures."""
    return list(BUILTIN_ARCHITECTURES.keys())
