"""Architecture dataclasses: Pipeline, Stage, ModelCall.

These describe *expected* (estimated) architectures before running them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
