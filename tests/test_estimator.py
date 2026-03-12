"""Tests for costmodel.estimator — CostEstimate, estimate().

Acceptance criteria from issue #7:
  - estimate(SINGLE_AGENT_OPUS) returns correct per_run_usd matching manual calculation.
  - estimate(THREE_AGENT_SONNET) costs ~3× more than estimate(SINGLE_AGENT_SONNET).
"""

from __future__ import annotations

import pytest

from costmodel.estimator import (
    CostEstimate,
    estimate,
    estimate_from_task_type,
    estimate_from_tokens,
    load_architecture_from_yaml,
)
from costmodel.models import (
    Architecture,
    ModelCall,
    Stage,
    BUILTIN_ARCHITECTURES,
)


# ---------------------------------------------------------------------------
# Shared test architectures
# ---------------------------------------------------------------------------

# Single Opus agent — manual cost:
#   input=3000 × $15.00/MTok = $0.045000
#   output=400  × $75.00/MTok = $0.030000
#   total = $0.075000
SINGLE_AGENT_OPUS = BUILTIN_ARCHITECTURES["single-agent-opus"]

# Single Sonnet agent — manual cost:
#   input=3000 × $3.00/MTok = $0.009000
#   output=400  × $15.00/MTok = $0.006000
#   total = $0.015000
SINGLE_AGENT_SONNET = BUILTIN_ARCHITECTURES["single-agent-sonnet"]

# Three Sonnet agents running in parallel — each does exactly what
# SINGLE_AGENT_SONNET does, so cost = 3 × SINGLE_AGENT_SONNET = $0.045000
THREE_AGENT_SONNET = Architecture(
    name="three-agent-sonnet",
    description="3 parallel Sonnet agents, each equivalent to single-agent-sonnet",
    stages=[
        Stage(
            name="main",
            parallel=True,
            calls=[
                ModelCall(
                    "claude-sonnet-4-5",
                    input_tokens=3000,
                    output_tokens=400,
                    repeats=3,
                )
            ],
        )
    ],
)


# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------

class TestCostEstimateDataclass:
    """Verify the CostEstimate structure matches the issue spec."""

    def test_has_required_fields(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert hasattr(est, "architecture_name")
        assert hasattr(est, "per_run_usd")
        assert hasattr(est, "per_1000_runs_usd")
        assert hasattr(est, "per_stage")
        assert hasattr(est, "token_breakdown")
        assert hasattr(est, "latency_seconds")
        assert hasattr(est, "confidence")

    def test_per_stage_is_dict(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert isinstance(est.per_stage, dict)

    def test_token_breakdown_is_dict(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert isinstance(est.token_breakdown, dict)

    def test_token_breakdown_has_expected_keys(self):
        est = estimate(SINGLE_AGENT_OPUS)
        for stage_key, breakdown in est.token_breakdown.items():
            assert "input" in breakdown, f"Stage '{stage_key}' missing 'input' key"
            assert "output" in breakdown, f"Stage '{stage_key}' missing 'output' key"
            assert "cache" in breakdown, f"Stage '{stage_key}' missing 'cache' key"

    def test_latency_seconds_is_float(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert isinstance(est.latency_seconds, float)
        assert est.latency_seconds >= 0.0

    def test_confidence_valid_values(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.confidence in {"high", "medium", "low"}

    def test_per_1000_is_1000x_per_run(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.per_1000_runs_usd == pytest.approx(est.per_run_usd * 1000, rel=1e-9)

    def test_architecture_name_matches(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.architecture_name == SINGLE_AGENT_OPUS.name


# ---------------------------------------------------------------------------
# Per-stage cost breakdown
# ---------------------------------------------------------------------------

class TestPerStageCost:
    """Verify per_stage costs are computed correctly."""

    def test_single_stage_key_present(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert "main" in est.per_stage

    def test_single_stage_cost_matches_total(self):
        est = estimate(SINGLE_AGENT_OPUS)
        stage_sum = sum(est.per_stage.values())
        assert stage_sum == pytest.approx(est.per_run_usd, rel=1e-9)

    def test_multi_stage_keys_present(self):
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)
        assert "planning" in est.per_stage
        assert "review" in est.per_stage
        assert "synthesis" in est.per_stage

    def test_multi_stage_costs_sum_to_total(self):
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)
        stage_sum = sum(est.per_stage.values())
        assert stage_sum == pytest.approx(est.per_run_usd, rel=1e-9)

    def test_conditional_stage_cost_scaled(self):
        """Stages with condition < 1.0 have their cost scaled accordingly."""
        arch = Architecture(
            name="cond-test",
            stages=[
                Stage(
                    name="always",
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=100)],
                    condition=1.0,
                ),
                Stage(
                    name="half",
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=100)],
                    condition=0.5,
                ),
            ],
        )
        est = estimate(arch)
        # 'half' stage cost should be exactly 0.5× 'always' stage cost
        assert est.per_stage["half"] == pytest.approx(
            est.per_stage["always"] * 0.5, rel=1e-9
        )

    def test_repeats_multiply_stage_cost(self):
        """A call with repeats=3 should cost 3× a single call."""
        arch_single = Architecture(
            name="single",
            stages=[
                Stage(
                    name="main",
                    calls=[ModelCall("claude-haiku-4-5", input_tokens=500, output_tokens=100)],
                )
            ],
        )
        arch_triple = Architecture(
            name="triple",
            stages=[
                Stage(
                    name="main",
                    calls=[ModelCall("claude-haiku-4-5", input_tokens=500, output_tokens=100, repeats=3)],
                )
            ],
        )
        est_single = estimate(arch_single)
        est_triple = estimate(arch_triple)
        assert est_triple.per_run_usd == pytest.approx(
            est_single.per_run_usd * 3, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Token breakdown
# ---------------------------------------------------------------------------

class TestTokenBreakdown:
    """Verify token_breakdown structure and values."""

    def test_single_stage_token_breakdown(self):
        est = estimate(SINGLE_AGENT_OPUS)
        breakdown = est.token_breakdown["main"]
        assert breakdown["input"] == 3000
        assert breakdown["output"] == 400
        assert breakdown["cache"] == 0

    def test_multi_stage_breakdown_correct(self):
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)

        # planning: opus, input=2000, output=500
        assert est.token_breakdown["planning"]["input"] == 2000
        assert est.token_breakdown["planning"]["output"] == 500

        # review: sonnet, input=8000, output=1000, repeats=3 → ×3
        assert est.token_breakdown["review"]["input"] == 24000
        assert est.token_breakdown["review"]["output"] == 3000

        # synthesis: opus, input=5000, output=800
        assert est.token_breakdown["synthesis"]["input"] == 5000
        assert est.token_breakdown["synthesis"]["output"] == 800

    def test_cache_tokens_included(self):
        arch = Architecture(
            name="cache-arch",
            stages=[
                Stage(
                    name="main",
                    calls=[
                        ModelCall(
                            "claude-sonnet-4-5",
                            input_tokens=2000,
                            output_tokens=300,
                            cached_input_tokens=1000,
                            cache_write_tokens=500,
                        )
                    ],
                )
            ],
        )
        est = estimate(arch)
        # cache = cached_input_tokens + cache_write_tokens = 1000 + 500 = 1500
        assert est.token_breakdown["main"]["cache"] == 1500

    def test_conditional_tokens_scaled(self):
        """Tokens in conditional stages are scaled by condition probability."""
        arch = Architecture(
            name="cond-tokens",
            stages=[
                Stage(
                    name="sometimes",
                    condition=0.5,
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=200)],
                )
            ],
        )
        est = estimate(arch)
        # 50% chance → scaled tokens
        assert est.token_breakdown["sometimes"]["input"] == 500
        assert est.token_breakdown["sometimes"]["output"] == 100

    def test_total_input_tokens_property(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.total_input_tokens == 3000

    def test_total_output_tokens_property(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.total_output_tokens == 400

    def test_total_tokens_property(self):
        est = estimate(SINGLE_AGENT_OPUS)
        # No cache, so total = input + output = 3400
        assert est.total_tokens == 3400


# ---------------------------------------------------------------------------
# Latency estimate
# ---------------------------------------------------------------------------

class TestLatency:
    """Verify latency_seconds is computed correctly (50 tok/s approximation)."""

    def test_single_call_latency(self):
        # output_tokens=400, latency = 400/50 = 8.0 seconds
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.latency_seconds == pytest.approx(8.0, rel=1e-9)

    def test_sequential_stages_sum(self):
        """Sequential stages: latency = sum of all call latencies."""
        arch = Architecture(
            name="seq-two",
            stages=[
                Stage(
                    name="stage-a",
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=100)],
                ),
                Stage(
                    name="stage-b",
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=200)],
                ),
            ],
        )
        est = estimate(arch)
        # stage-a: 100/50 = 2s; stage-b: 200/50 = 4s; total = 6s
        assert est.latency_seconds == pytest.approx(6.0, rel=1e-9)

    def test_parallel_stage_uses_max(self):
        """Parallel stages: latency = max of concurrent call latencies."""
        arch = Architecture(
            name="par-stage",
            stages=[
                Stage(
                    name="parallel-work",
                    parallel=True,
                    calls=[
                        ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=200),
                        ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=600),
                    ],
                )
            ],
        )
        est = estimate(arch)
        # call-1: 200/50 = 4s; call-2: 600/50 = 12s; parallel → max = 12s
        assert est.latency_seconds == pytest.approx(12.0, rel=1e-9)

    def test_parallel_repeats_use_max(self):
        """For parallel stages, repeats are concurrent — max, not sum."""
        arch = Architecture(
            name="par-repeats",
            stages=[
                Stage(
                    name="agents",
                    parallel=True,
                    calls=[
                        ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=400, repeats=3)
                    ],
                )
            ],
        )
        est = estimate(arch)
        # each repeat: 400/50 = 8s; parallel → max(8, 8, 8) = 8s (not 24s)
        assert est.latency_seconds == pytest.approx(8.0, rel=1e-9)

    def test_sequential_repeats_sum(self):
        """For sequential stages, repeats are additive."""
        arch = Architecture(
            name="seq-repeats",
            stages=[
                Stage(
                    name="loop",
                    parallel=False,
                    calls=[
                        ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=400, repeats=3)
                    ],
                )
            ],
        )
        est = estimate(arch)
        # each repeat: 400/50 = 8s; sequential → 3 × 8s = 24s
        assert est.latency_seconds == pytest.approx(24.0, rel=1e-9)

    def test_mixed_sequential_and_parallel_stages(self):
        """Overall latency = sum of stage latencies (stages always sequential)."""
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)
        # planning: opus, output=500 → 500/50 = 10s (sequential stage)
        # review: sonnet, output=1000, repeats=3, parallel → max(20,20,20) = 20s
        # synthesis: opus, output=800 → 800/50 = 16s (sequential stage)
        # total = 10 + 20 + 16 = 46s
        assert est.latency_seconds == pytest.approx(46.0, rel=1e-9)

    def test_latency_not_affected_by_condition(self):
        """Condition probability does not reduce latency — if it runs, it takes full time."""
        arch = Architecture(
            name="cond-latency",
            stages=[
                Stage(
                    name="sometimes",
                    condition=0.1,
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=500)],
                )
            ],
        )
        est = estimate(arch)
        # latency = 500/50 = 10s regardless of condition=0.1
        assert est.latency_seconds == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------

class TestConfidence:
    """Verify confidence computation."""

    def test_high_when_all_explicit(self):
        """'high' when all token counts are explicit (no heuristics, no conditions, no cache)."""
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.confidence == "high"

    def test_medium_when_conditional(self):
        """'medium' when stages have condition < 1.0."""
        arch = Architecture(
            name="cond",
            stages=[
                Stage(
                    name="maybe",
                    condition=0.5,
                    calls=[ModelCall("claude-sonnet-4-5", input_tokens=1000, output_tokens=200)],
                )
            ],
        )
        est = estimate(arch)
        assert est.confidence == "medium"

    def test_medium_when_cache_used(self):
        """'medium' when cache tokens are specified (actual hit rate unknown)."""
        arch = Architecture(
            name="cache",
            stages=[
                Stage(
                    name="main",
                    calls=[
                        ModelCall(
                            "claude-sonnet-4-5",
                            input_tokens=1000,
                            output_tokens=200,
                            cached_input_tokens=500,
                        )
                    ],
                )
            ],
        )
        est = estimate(arch)
        assert est.confidence == "medium"

    def test_low_when_heuristics_used(self):
        """'low' when token counts come from task-type heuristics."""
        est = estimate_from_task_type("claude-sonnet-4-5", "code_review")
        assert est.confidence == "low"

    def test_estimate_from_tokens_is_high(self):
        """estimate_from_tokens() uses explicit counts → confidence='high'."""
        est = estimate_from_tokens("claude-sonnet-4-5", 3000, 400)
        assert est.confidence == "high"


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERIA — Issue #7
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """End-to-end scenarios matching the issue acceptance criteria."""

    def test_single_agent_opus_correct_per_run_usd(self):
        """estimate(SINGLE_AGENT_OPUS) per_run_usd matches manual calculation.

        Manual calculation:
          model = claude-opus-4-5
          input_tokens = 3000  @ $15.00 / MTok = $0.045000
          output_tokens = 400  @ $75.00 / MTok = $0.030000
          total = $0.075000
        """
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.per_run_usd == pytest.approx(0.075000, abs=1e-9), (
            f"Expected $0.075000, got ${est.per_run_usd}"
        )

    def test_single_agent_opus_per_1000_runs(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.per_1000_runs_usd == pytest.approx(75.000, abs=1e-6), (
            f"Expected $75.00 per 1000 runs, got ${est.per_1000_runs_usd}"
        )

    def test_single_agent_opus_token_breakdown(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.token_breakdown["main"]["input"] == 3000
        assert est.token_breakdown["main"]["output"] == 400
        assert est.token_breakdown["main"]["cache"] == 0

    def test_single_agent_opus_high_confidence(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert est.confidence == "high"

    def test_three_agent_sonnet_costs_3x_single_agent_sonnet(self):
        """estimate(THREE_AGENT_SONNET) costs ~3× more than estimate(SINGLE_AGENT_SONNET).

        THREE_AGENT_SONNET: 3 parallel Sonnet agents each doing the same work
        as SINGLE_AGENT_SONNET (3000 input, 400 output).
          single:  3000*3/1e6 + 400*15/1e6 = 0.009 + 0.006 = $0.015000
          three:   same × 3 = $0.045000
          ratio:   3.00×
        """
        est_single = estimate(SINGLE_AGENT_SONNET)
        est_three = estimate(THREE_AGENT_SONNET)

        ratio = est_three.per_run_usd / est_single.per_run_usd
        assert ratio == pytest.approx(3.0, rel=1e-9), (
            f"Expected ~3× ratio, got {ratio:.4f}× "
            f"(single=${est_single.per_run_usd:.6f}, "
            f"three=${est_three.per_run_usd:.6f})"
        )

    def test_three_agent_sonnet_per_run_value(self):
        """THREE_AGENT_SONNET absolute cost = $0.045."""
        est = estimate(THREE_AGENT_SONNET)
        assert est.per_run_usd == pytest.approx(0.045000, abs=1e-9)

    def test_single_agent_sonnet_per_run_value(self):
        """SINGLE_AGENT_SONNET absolute cost = $0.015."""
        est = estimate(SINGLE_AGENT_SONNET)
        assert est.per_run_usd == pytest.approx(0.015000, abs=1e-9)

    def test_three_agent_sonnet_parallel_latency(self):
        """THREE_AGENT_SONNET latency = single call latency (not 3×, because parallel)."""
        est_single = estimate(SINGLE_AGENT_SONNET)
        est_three = estimate(THREE_AGENT_SONNET)
        # Both have output_tokens=400, latency=8s for a single invocation
        # Parallel execution → max(8, 8, 8) = 8s (same as single agent)
        assert est_three.latency_seconds == pytest.approx(est_single.latency_seconds, rel=1e-9)

    def test_return_type_is_cost_estimate(self):
        est = estimate(SINGLE_AGENT_OPUS)
        assert isinstance(est, CostEstimate)


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for robustness."""

    def test_empty_stages(self):
        arch = Architecture(name="empty", stages=[])
        est = estimate(arch)
        assert est.per_run_usd == 0.0
        assert est.latency_seconds == 0.0
        assert est.per_stage == {}
        assert est.token_breakdown == {}

    def test_zero_output_tokens_zero_latency(self):
        arch = Architecture(
            name="no-output",
            stages=[
                Stage(
                    name="main",
                    calls=[ModelCall("claude-haiku-4-5", input_tokens=100, output_tokens=0)],
                )
            ],
        )
        est = estimate(arch)
        assert est.latency_seconds == pytest.approx(0.0, abs=1e-9)

    def test_duplicate_stage_names_deduplicated(self):
        """Two stages with the same name should get distinct keys."""
        arch = Architecture(
            name="dup-names",
            stages=[
                Stage(
                    name="main",
                    calls=[ModelCall("claude-haiku-4-5", input_tokens=100, output_tokens=100)],
                ),
                Stage(
                    name="main",
                    calls=[ModelCall("claude-haiku-4-5", input_tokens=100, output_tokens=100)],
                ),
            ],
        )
        est = estimate(arch)
        # Both stages should be present
        assert len(est.per_stage) == 2
        assert "main" in est.per_stage
        assert "main_2" in est.per_stage

    def test_condition_zero_stage_zero_cost(self):
        """A stage with condition=0.0 contributes zero cost."""
        arch = Architecture(
            name="never-runs",
            stages=[
                Stage(
                    name="never",
                    condition=0.0,
                    calls=[ModelCall("claude-opus-4-5", input_tokens=10000, output_tokens=5000)],
                )
            ],
        )
        est = estimate(arch)
        assert est.per_run_usd == pytest.approx(0.0, abs=1e-9)

    def test_all_builtin_architectures_estimate_without_error(self):
        """All built-in architectures should produce a valid estimate."""
        from costmodel.models import BUILTIN_ARCHITECTURES
        for name, arch in BUILTIN_ARCHITECTURES.items():
            est = estimate(arch)
            assert est.per_run_usd >= 0.0, f"{name}: per_run_usd is negative"
            assert est.latency_seconds >= 0.0, f"{name}: latency is negative"
            assert est.confidence in {"high", "medium", "low"}, (
                f"{name}: invalid confidence '{est.confidence}'"
            )

    def test_per_stage_dict_values_are_floats(self):
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)
        for stage_name, cost in est.per_stage.items():
            assert isinstance(cost, float), f"Stage '{stage_name}': cost is not float"

    def test_token_breakdown_values_are_ints(self):
        arch = BUILTIN_ARCHITECTURES["3-agent-sonnet"]
        est = estimate(arch)
        for stage_name, breakdown in est.token_breakdown.items():
            for key in ("input", "output", "cache"):
                val = breakdown[key]
                assert isinstance(val, int), (
                    f"Stage '{stage_name}' token['{key}'] = {val!r} is not int"
                )


# ---------------------------------------------------------------------------
# estimate_from_tokens (convenience helper)
# ---------------------------------------------------------------------------

class TestEstimateFromTokens:
    def test_basic(self):
        est = estimate_from_tokens("claude-opus-4-5", 3000, 400)
        assert est.per_run_usd == pytest.approx(0.075000, abs=1e-9)

    def test_with_cached_input(self):
        # Cached input uses cache_read rate ($1.50/MTok for opus)
        est = estimate_from_tokens("claude-opus-4-5", 1000, 200, cached_input_tokens=2000)
        # input: 1000*15/1e6=0.015
        # output: 200*75/1e6=0.015
        # cache_read: 2000*1.50/1e6=0.003
        # total = 0.033
        assert est.per_run_usd == pytest.approx(0.033000, abs=1e-9)

    def test_returns_cost_estimate(self):
        est = estimate_from_tokens("claude-haiku-4-5", 100, 100)
        assert isinstance(est, CostEstimate)

    def test_confidence_is_high(self):
        est = estimate_from_tokens("claude-haiku-4-5", 100, 100)
        assert est.confidence == "high"


# ---------------------------------------------------------------------------
# estimate_from_task_type (convenience helper)
# ---------------------------------------------------------------------------

class TestEstimateFromTaskType:
    def test_returns_cost_estimate(self):
        est = estimate_from_task_type("claude-haiku-4-5", "code_review")
        assert isinstance(est, CostEstimate)

    def test_confidence_is_low(self):
        est = estimate_from_task_type("claude-haiku-4-5", "code_review")
        assert est.confidence == "low"

    def test_notes_mention_heuristics(self):
        est = estimate_from_task_type("claude-haiku-4-5", "code_review")
        notes_combined = " ".join(est.notes).lower()
        assert "heuristic" in notes_combined

    def test_positive_cost(self):
        est = estimate_from_task_type("claude-sonnet-4-5", "planning")
        assert est.per_run_usd > 0.0


# ---------------------------------------------------------------------------
# load_architecture_from_yaml (YAML loader)
# ---------------------------------------------------------------------------

class TestLoadArchitectureFromYaml:
    def test_load_and_estimate_single_stage(self, tmp_path):
        yaml_content = """\
name: test-arch
description: Test architecture
stages:
  - name: main
    calls:
      - model: claude-haiku-4-5
        input_tokens: 1000
        output_tokens: 200
"""
        f = tmp_path / "arch.yaml"
        f.write_text(yaml_content)
        arch = load_architecture_from_yaml(f)
        est = estimate(arch)
        assert est.architecture_name == "test-arch"
        # haiku: 1000*0.80/1e6 + 200*4.00/1e6 = 0.0008 + 0.0008 = 0.0016
        assert est.per_run_usd == pytest.approx(0.001600, abs=1e-9)
        assert est.confidence == "high"

    def test_load_parallel_stage(self, tmp_path):
        yaml_content = """\
name: par-arch
stages:
  - name: agents
    parallel: true
    calls:
      - model: claude-sonnet-4-5
        input_tokens: 3000
        output_tokens: 400
        repeats: 3
"""
        f = tmp_path / "arch.yaml"
        f.write_text(yaml_content)
        arch = load_architecture_from_yaml(f)
        est = estimate(arch)
        # 3 parallel agents, each = single-agent-sonnet ($0.015) → total $0.045
        assert est.per_run_usd == pytest.approx(0.045000, abs=1e-9)
        # parallel latency = max(8, 8, 8) = 8s
        assert est.latency_seconds == pytest.approx(8.0, rel=1e-9)
