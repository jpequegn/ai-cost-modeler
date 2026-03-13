"""Tests for Architecture dataclasses, presets, and YAML round-trip.

Acceptance criteria from issue #9:
  - All 5 presets instantiate without error.
  - YAML round-trip: save and load produces identical architecture.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import fields

import pytest
import yaml

from costmodel.architectures.presets import (
    ANTHROPIC_CODE_REVIEW,
    SINGLE_AGENT_HAIKU,
    SINGLE_AGENT_OPUS,
    SINGLE_AGENT_SONNET,
    THREE_AGENT_SONNET,
)
from costmodel.estimator import estimate
from costmodel.models import Architecture, ModelCall, Stage


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------

class TestModelCallDataclass:
    """Verify ModelCall has exactly the fields specified in the issue."""

    def test_required_fields_exist(self):
        call = ModelCall(model="claude-sonnet-4-5", input_tokens=1000, output_tokens=200)
        assert call.model == "claude-sonnet-4-5"
        assert call.input_tokens == 1000
        assert call.output_tokens == 200

    def test_optional_fields_have_defaults(self):
        call = ModelCall(model="claude-sonnet-4-5", input_tokens=1000, output_tokens=200)
        assert call.cached_input_tokens == 0
        assert call.repeats == 1

    def test_all_fields_settable(self):
        call = ModelCall(
            model="claude-opus-4-5",
            input_tokens=5000,
            output_tokens=800,
            cached_input_tokens=1000,
            repeats=3,
        )
        assert call.cached_input_tokens == 1000
        assert call.repeats == 3

    def test_is_dataclass(self):
        call = ModelCall(model="claude-haiku-4-5", input_tokens=100, output_tokens=50)
        # dataclasses expose __dataclass_fields__
        assert hasattr(call, "__dataclass_fields__")


class TestStageDataclass:
    """Verify Stage has the fields and defaults specified in the issue."""

    def test_required_fields(self):
        stage = Stage(
            name="main",
            calls=[ModelCall("claude-sonnet-4-5", 1000, 200)],
        )
        assert stage.name == "main"
        assert len(stage.calls) == 1

    def test_parallel_default_false(self):
        stage = Stage(name="s", calls=[])
        assert stage.parallel is False

    def test_condition_default_one(self):
        stage = Stage(name="s", calls=[])
        assert stage.condition == 1.0

    def test_parallel_settable(self):
        stage = Stage(name="s", calls=[], parallel=True)
        assert stage.parallel is True

    def test_condition_settable(self):
        stage = Stage(name="s", calls=[], condition=0.5)
        assert stage.condition == 0.5

    def test_is_dataclass(self):
        stage = Stage(name="s", calls=[])
        assert hasattr(stage, "__dataclass_fields__")


class TestArchitectureDataclass:
    """Verify Architecture has the fields specified in the issue."""

    def test_required_fields(self):
        arch = Architecture(name="test", stages=[])
        assert arch.name == "test"
        assert arch.stages == []

    def test_description_default_empty(self):
        arch = Architecture(name="test", stages=[])
        assert arch.description == ""

    def test_description_settable(self):
        arch = Architecture(name="test", stages=[], description="hello")
        assert arch.description == "hello"

    def test_is_dataclass(self):
        arch = Architecture(name="test", stages=[])
        assert hasattr(arch, "__dataclass_fields__")

    def test_has_from_yaml_classmethod(self):
        assert callable(getattr(Architecture, "from_yaml", None))

    def test_has_to_yaml_method(self):
        arch = Architecture(name="test", stages=[])
        assert callable(getattr(arch, "to_yaml", None))


# ---------------------------------------------------------------------------
# Preset instantiation
# ---------------------------------------------------------------------------

class TestPresetInstantiation:
    """All 5 built-in presets must instantiate without error."""

    @pytest.mark.parametrize("arch", [
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    ], ids=[
        "SINGLE_AGENT_HAIKU",
        "SINGLE_AGENT_SONNET",
        "SINGLE_AGENT_OPUS",
        "THREE_AGENT_SONNET",
        "ANTHROPIC_CODE_REVIEW",
    ])
    def test_is_architecture_instance(self, arch):
        assert isinstance(arch, Architecture)

    @pytest.mark.parametrize("arch", [
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    ], ids=[
        "SINGLE_AGENT_HAIKU",
        "SINGLE_AGENT_SONNET",
        "SINGLE_AGENT_OPUS",
        "THREE_AGENT_SONNET",
        "ANTHROPIC_CODE_REVIEW",
    ])
    def test_has_name(self, arch):
        assert isinstance(arch.name, str) and arch.name

    @pytest.mark.parametrize("arch", [
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    ], ids=[
        "SINGLE_AGENT_HAIKU",
        "SINGLE_AGENT_SONNET",
        "SINGLE_AGENT_OPUS",
        "THREE_AGENT_SONNET",
        "ANTHROPIC_CODE_REVIEW",
    ])
    def test_has_at_least_one_stage(self, arch):
        assert len(arch.stages) >= 1

    def test_single_agent_haiku_name(self):
        assert SINGLE_AGENT_HAIKU.name == "single-agent-haiku"

    def test_single_agent_sonnet_name(self):
        assert SINGLE_AGENT_SONNET.name == "single-agent-sonnet"

    def test_single_agent_opus_name(self):
        assert SINGLE_AGENT_OPUS.name == "single-agent-opus"

    def test_three_agent_sonnet_name(self):
        assert THREE_AGENT_SONNET.name == "3-agent-sonnet-parallel"

    def test_anthropic_code_review_name(self):
        assert ANTHROPIC_CODE_REVIEW.name == "anthropic-code-review"

    def test_three_agent_sonnet_has_parallel_stage(self):
        """THREE_AGENT_SONNET must have at least one parallel stage."""
        has_parallel = any(s.parallel for s in THREE_AGENT_SONNET.stages)
        assert has_parallel, "THREE_AGENT_SONNET has no parallel stage"

    def test_three_agent_sonnet_has_three_stages(self):
        assert len(THREE_AGENT_SONNET.stages) == 3

    def test_three_agent_sonnet_stage_names(self):
        names = [s.name for s in THREE_AGENT_SONNET.stages]
        assert "planning" in names
        assert "review" in names
        assert "synthesis" in names

    def test_anthropic_code_review_uses_external_model(self):
        """ANTHROPIC_CODE_REVIEW should use the 'external' placeholder model."""
        all_models = [
            call.model
            for stage in ANTHROPIC_CODE_REVIEW.stages
            for call in stage.calls
        ]
        assert "external" in all_models

    def test_anthropic_code_review_has_flat_rate_description(self):
        """Description should mention the flat-rate pricing."""
        desc_lower = ANTHROPIC_CODE_REVIEW.description.lower()
        assert "25" in desc_lower or "flat" in desc_lower or "rate" in desc_lower

    def test_single_agent_presets_use_correct_models(self):
        haiku_model = SINGLE_AGENT_HAIKU.stages[0].calls[0].model
        sonnet_model = SINGLE_AGENT_SONNET.stages[0].calls[0].model
        opus_model = SINGLE_AGENT_OPUS.stages[0].calls[0].model
        assert "haiku" in haiku_model.lower()
        assert "sonnet" in sonnet_model.lower()
        assert "opus" in opus_model.lower()


# ---------------------------------------------------------------------------
# Preset estimation (all produce non-negative costs)
# ---------------------------------------------------------------------------

class TestPresetEstimation:
    """All presets should estimate without error."""

    @pytest.mark.parametrize("arch", [
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    ], ids=[
        "SINGLE_AGENT_HAIKU",
        "SINGLE_AGENT_SONNET",
        "SINGLE_AGENT_OPUS",
        "THREE_AGENT_SONNET",
        "ANTHROPIC_CODE_REVIEW",
    ])
    def test_estimate_runs_without_error(self, arch):
        est = estimate(arch)
        assert est.per_run_usd >= 0.0
        assert est.latency_seconds >= 0.0

    def test_haiku_cheaper_than_sonnet(self):
        est_haiku = estimate(SINGLE_AGENT_HAIKU)
        est_sonnet = estimate(SINGLE_AGENT_SONNET)
        assert est_haiku.per_run_usd < est_sonnet.per_run_usd

    def test_sonnet_cheaper_than_opus(self):
        est_sonnet = estimate(SINGLE_AGENT_SONNET)
        est_opus = estimate(SINGLE_AGENT_OPUS)
        assert est_sonnet.per_run_usd < est_opus.per_run_usd

    def test_three_agent_sonnet_more_expensive_than_single(self):
        est_single = estimate(SINGLE_AGENT_SONNET)
        est_three = estimate(THREE_AGENT_SONNET)
        assert est_three.per_run_usd > est_single.per_run_usd

    def test_anthropic_code_review_zero_cost(self):
        """Flat-rate service has zero token-based cost (billing not modelled)."""
        est = estimate(ANTHROPIC_CODE_REVIEW)
        assert est.per_run_usd == pytest.approx(0.0, abs=1e-9)

    def test_three_agent_sonnet_parallel_latency(self):
        """Parallel review stage means latency < sequential would be."""
        est = estimate(THREE_AGENT_SONNET)
        # review stage has repeats=3 parallel — latency should not be 3×
        # check latency is reasonable (< 3 × output_tok/50 for 3 repeats)
        # planning=400/50=8s, review_parallel=800/50=16s, synthesis=700/50=14s
        # total = 8+16+14 = 38s (not 8+48+14 = 70s sequential)
        assert est.latency_seconds < 70.0


# ---------------------------------------------------------------------------
# Architecture.from_yaml / to_yaml — YAML round-trip
# ---------------------------------------------------------------------------

class TestYamlRoundTrip:
    """YAML round-trip: save and load produces identical architecture."""

    @pytest.mark.parametrize("arch", [
        SINGLE_AGENT_HAIKU,
        SINGLE_AGENT_SONNET,
        SINGLE_AGENT_OPUS,
        THREE_AGENT_SONNET,
        ANTHROPIC_CODE_REVIEW,
    ], ids=[
        "SINGLE_AGENT_HAIKU",
        "SINGLE_AGENT_SONNET",
        "SINGLE_AGENT_OPUS",
        "THREE_AGENT_SONNET",
        "ANTHROPIC_CODE_REVIEW",
    ])
    def test_round_trip_identity(self, arch, tmp_path):
        """to_yaml → from_yaml produces identical architecture."""
        yaml_file = tmp_path / f"{arch.name}.yaml"
        arch.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)

        assert loaded.name == arch.name
        assert loaded.description == arch.description
        assert loaded.tags == arch.tags
        assert len(loaded.stages) == len(arch.stages)

        for s_orig, s_loaded in zip(arch.stages, loaded.stages):
            assert s_loaded.name == s_orig.name
            assert s_loaded.parallel == s_orig.parallel
            assert s_loaded.condition == pytest.approx(s_orig.condition)
            assert len(s_loaded.calls) == len(s_orig.calls)
            for c_orig, c_loaded in zip(s_orig.calls, s_loaded.calls):
                assert c_loaded.model == c_orig.model
                assert c_loaded.input_tokens == c_orig.input_tokens
                assert c_loaded.output_tokens == c_orig.output_tokens
                assert c_loaded.cached_input_tokens == c_orig.cached_input_tokens
                assert c_loaded.cache_write_tokens == c_orig.cache_write_tokens
                assert c_loaded.repeats == c_orig.repeats

    def test_to_yaml_returns_string(self):
        arch = SINGLE_AGENT_SONNET
        result = arch.to_yaml()
        assert isinstance(result, str)
        assert "name:" in result
        assert "stages:" in result

    def test_to_yaml_writes_file(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        SINGLE_AGENT_SONNET.to_yaml(yaml_file)
        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert "single-agent-sonnet" in content

    def test_from_yaml_produces_architecture_instance(self, tmp_path):
        yaml_file = tmp_path / "arch.yaml"
        SINGLE_AGENT_HAIKU.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)
        assert isinstance(loaded, Architecture)

    def test_from_yaml_string_path(self, tmp_path):
        """from_yaml should accept a str path, not just Path."""
        yaml_file = tmp_path / "arch.yaml"
        SINGLE_AGENT_OPUS.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(str(yaml_file))
        assert loaded.name == SINGLE_AGENT_OPUS.name

    def test_yaml_is_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "arch.yaml"
        text = THREE_AGENT_SONNET.to_yaml(yaml_file)
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "stages" in parsed

    def test_parallel_flag_preserved(self, tmp_path):
        yaml_file = tmp_path / "arch.yaml"
        THREE_AGENT_SONNET.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)
        review_stage = next(s for s in loaded.stages if s.name == "review")
        assert review_stage.parallel is True

    def test_repeats_preserved(self, tmp_path):
        yaml_file = tmp_path / "arch.yaml"
        THREE_AGENT_SONNET.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)
        review_stage = next(s for s in loaded.stages if s.name == "review")
        assert review_stage.calls[0].repeats == 3

    def test_condition_preserved(self, tmp_path):
        arch = Architecture(
            name="conditional",
            stages=[
                Stage(
                    name="maybe",
                    condition=0.7,
                    calls=[ModelCall("claude-sonnet-4-5", 1000, 200)],
                )
            ],
        )
        yaml_file = tmp_path / "arch.yaml"
        arch.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)
        assert loaded.stages[0].condition == pytest.approx(0.7)

    def test_cached_tokens_preserved(self, tmp_path):
        arch = Architecture(
            name="cached",
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
        yaml_file = tmp_path / "arch.yaml"
        arch.to_yaml(yaml_file)
        loaded = Architecture.from_yaml(yaml_file)
        call = loaded.stages[0].calls[0]
        assert call.cached_input_tokens == 1000
        assert call.cache_write_tokens == 500

    def test_round_trip_estimates_match(self, tmp_path):
        """Estimates from original and loaded architectures must be identical."""
        for arch in [SINGLE_AGENT_HAIKU, THREE_AGENT_SONNET]:
            yaml_file = tmp_path / f"{arch.name}.yaml"
            arch.to_yaml(yaml_file)
            loaded = Architecture.from_yaml(yaml_file)

            est_orig = estimate(arch)
            est_loaded = estimate(loaded)

            assert est_loaded.per_run_usd == pytest.approx(est_orig.per_run_usd, rel=1e-9)
            assert est_loaded.latency_seconds == pytest.approx(est_orig.latency_seconds, rel=1e-9)

    def test_from_yaml_manual_file(self, tmp_path):
        """from_yaml works with a manually-written YAML (not from to_yaml)."""
        yaml_content = """\
name: manual-arch
description: Manually written test architecture
stages:
  - name: planning
    calls:
      - model: claude-opus-4-5
        input_tokens: 2000
        output_tokens: 500
  - name: review
    parallel: true
    condition: 0.9
    calls:
      - model: claude-sonnet-4-5
        input_tokens: 8000
        output_tokens: 1000
        repeats: 2
"""
        yaml_file = tmp_path / "manual.yaml"
        yaml_file.write_text(yaml_content)
        arch = Architecture.from_yaml(yaml_file)

        assert arch.name == "manual-arch"
        assert arch.description == "Manually written test architecture"
        assert len(arch.stages) == 2
        assert arch.stages[1].parallel is True
        assert arch.stages[1].condition == pytest.approx(0.9)
        assert arch.stages[1].calls[0].repeats == 2


# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------

class TestImports:
    """Verify the public API is importable from the expected locations."""

    def test_import_from_architectures_package(self):
        from costmodel.architectures import (
            SINGLE_AGENT_HAIKU,
            SINGLE_AGENT_SONNET,
            SINGLE_AGENT_OPUS,
            THREE_AGENT_SONNET,
            ANTHROPIC_CODE_REVIEW,
        )
        assert SINGLE_AGENT_HAIKU is not None
        assert SINGLE_AGENT_SONNET is not None
        assert SINGLE_AGENT_OPUS is not None
        assert THREE_AGENT_SONNET is not None
        assert ANTHROPIC_CODE_REVIEW is not None

    def test_import_from_presets_directly(self):
        from costmodel.architectures.presets import (
            SINGLE_AGENT_HAIKU,
            SINGLE_AGENT_SONNET,
            SINGLE_AGENT_OPUS,
            THREE_AGENT_SONNET,
            ANTHROPIC_CODE_REVIEW,
        )
        assert isinstance(SINGLE_AGENT_HAIKU, Architecture)

    def test_architecture_from_yaml_importable_from_models(self):
        from costmodel.models import Architecture
        assert hasattr(Architecture, "from_yaml")
