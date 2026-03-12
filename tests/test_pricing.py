"""Unit tests for costmodel.pricing — cost_usd, count_tokens, estimate_output_tokens."""

from __future__ import annotations

import pytest
import tiktoken

from costmodel.pricing import (
    PRICING,
    PRICING_VERSION,
    MODEL_ALIASES,
    cost_usd,
    cost_for_call,
    count_tokens,
    estimate_output_tokens,
    resolve_model,
    list_models,
)


# ---------------------------------------------------------------------------
# PRICING dict structure
# ---------------------------------------------------------------------------

class TestPricingTable:
    """Verify the PRICING dict is well-formed."""

    REQUIRED_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    REQUIRED_KEYS = {"input", "output", "cache_write", "cache_read"}

    def test_required_models_present(self):
        for model in self.REQUIRED_MODELS:
            assert model in PRICING, f"Missing model: {model}"

    def test_all_models_have_required_price_keys(self):
        for model, prices in PRICING.items():
            for key in self.REQUIRED_KEYS:
                assert key in prices, f"Model '{model}' missing price key '{key}'"

    def test_all_price_values_are_non_negative(self):
        for model, prices in PRICING.items():
            for key, val in prices.items():
                assert val >= 0.0, f"Model '{model}' key '{key}' is negative: {val}"

    def test_pricing_version_is_set(self):
        assert PRICING_VERSION, "PRICING_VERSION must be a non-empty string"
        # Basic YYYY-MM-DD format check
        import re
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", PRICING_VERSION), (
            f"PRICING_VERSION '{PRICING_VERSION}' should be YYYY-MM-DD"
        )

    def test_claude_opus_4_6_prices(self):
        p = PRICING["claude-opus-4-6"]
        assert p["input"] == 15.00
        assert p["output"] == 75.00
        assert p["cache_write"] == 18.75
        assert p["cache_read"] == 1.50

    def test_claude_sonnet_4_6_prices(self):
        p = PRICING["claude-sonnet-4-6"]
        assert p["input"] == 3.00
        assert p["output"] == 15.00
        assert p["cache_write"] == 3.75
        assert p["cache_read"] == 0.30

    def test_claude_haiku_4_5_prices(self):
        p = PRICING["claude-haiku-4-5"]
        assert p["input"] == 0.80
        assert p["output"] == 4.00
        assert p["cache_write"] == 1.00
        assert p["cache_read"] == 0.08

    def test_gpt_4o_prices(self):
        p = PRICING["gpt-4o"]
        assert p["input"] == 2.50
        assert p["output"] == 10.00

    def test_gpt_4o_mini_prices(self):
        p = PRICING["gpt-4o-mini"]
        assert p["input"] == 0.15
        assert p["output"] == 0.60


# ---------------------------------------------------------------------------
# cost_usd
# ---------------------------------------------------------------------------

class TestCostUsd:
    """Verify cost_usd() returns correct values to 6 decimal places."""

    def test_claude_sonnet_4_6_basic(self):
        # input: 1000 * 3.00 / 1e6 = 0.003
        # output: 500 * 15.00 / 1e6 = 0.0075
        # total = 0.010500
        result = cost_usd("claude-sonnet-4-6", 1000, 500)
        assert result == pytest.approx(0.010500, abs=1e-9)

    def test_claude_opus_4_6_acceptance_criteria(self):
        """Issue acceptance criteria: cost_usd('claude-opus-4-6', 10000, 2000)."""
        # input: 10000 * 15.00 / 1e6 = 0.150000
        # output: 2000 * 75.00 / 1e6 = 0.150000
        # total = 0.300000
        result = cost_usd("claude-opus-4-6", 10000, 2000)
        assert result == pytest.approx(0.300000, abs=1e-9)

    def test_claude_haiku_4_5_basic(self):
        # input: 1000 * 0.80 / 1e6 = 0.0008
        # output: 200 * 4.00 / 1e6 = 0.0008
        # total = 0.001600
        result = cost_usd("claude-haiku-4-5", 1000, 200)
        assert result == pytest.approx(0.001600, abs=1e-9)

    def test_gpt_4o_basic(self):
        # input: 1000 * 2.50 / 1e6 = 0.0025
        # output: 500 * 10.00 / 1e6 = 0.0050
        # total = 0.007500
        result = cost_usd("gpt-4o", 1000, 500)
        assert result == pytest.approx(0.007500, abs=1e-9)

    def test_gpt_4o_mini_basic(self):
        # input: 1000 * 0.15 / 1e6 = 0.000150
        # output: 500 * 0.60 / 1e6 = 0.000300
        # total = 0.000450
        result = cost_usd("gpt-4o-mini", 1000, 500)
        assert result == pytest.approx(0.000450, abs=1e-9)

    def test_with_cached_input(self):
        # claude-sonnet-4-6: input=1000, output=500, cached_input=2000
        # input: 1000 * 3.00 / 1e6  = 0.003000
        # output: 500 * 15.00 / 1e6 = 0.007500
        # cache_read: 2000 * 0.30 / 1e6 = 0.000600
        # total = 0.011100
        result = cost_usd("claude-sonnet-4-6", 1000, 500, cached_input=2000)
        assert result == pytest.approx(0.011100, abs=1e-9)

    def test_zero_tokens_returns_zero(self):
        assert cost_usd("claude-opus-4-6", 0, 0) == 0.0

    def test_alias_resolution_opus(self):
        direct = cost_usd("claude-opus-4-6", 1000, 500)
        aliased = cost_usd("opus", 1000, 500)
        assert direct == aliased

    def test_alias_resolution_sonnet(self):
        direct = cost_usd("claude-sonnet-4-6", 1000, 500)
        aliased = cost_usd("sonnet", 1000, 500)
        assert direct == aliased

    def test_alias_resolution_haiku(self):
        direct = cost_usd("claude-haiku-4-5", 1000, 500)
        aliased = cost_usd("haiku", 1000, 500)
        assert direct == aliased

    def test_result_has_six_decimal_precision(self):
        """Result is rounded to 6 decimal places."""
        result = cost_usd("claude-sonnet-4-6", 1, 1)
        # Should be expressible to 6 dp without extra floating-point noise
        assert result == round(result, 6)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            cost_usd("not-a-real-model-xyz", 100, 100)


# ---------------------------------------------------------------------------
# cost_for_call (extended signature — includes cache write)
# ---------------------------------------------------------------------------

class TestCostForCall:
    """Verify the extended cost_for_call() that includes cache_write_tokens."""

    def test_cache_write_included(self):
        # claude-sonnet-4-6: input=0, output=0, cached_input=0, cache_write=1000
        # cache_write: 1000 * 3.75 / 1e6 = 0.00375
        result = cost_for_call("claude-sonnet-4-6", 0, 0, cache_write_tokens=1000)
        assert result == pytest.approx(0.00375, abs=1e-9)

    def test_matches_cost_usd_when_no_cache_write(self):
        r1 = cost_for_call("claude-opus-4-6", 10000, 2000)
        r2 = cost_usd("claude-opus-4-6", 10000, 2000)
        assert r1 == pytest.approx(r2, abs=1e-9)

    def test_full_breakdown(self):
        # claude-sonnet-4-6:
        # input=1000 * 3.00/1e6 = 0.003
        # output=500 * 15.00/1e6 = 0.0075
        # cache_read=2000 * 0.30/1e6 = 0.0006
        # cache_write=500 * 3.75/1e6 = 0.001875
        # total = 0.012975
        result = cost_for_call(
            "claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
            cached_input_tokens=2000,
            cache_write_tokens=500,
        )
        assert result == pytest.approx(0.012975, abs=1e-9)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    """Verify count_tokens() matches tiktoken cl100k_base directly."""

    def _tiktoken_count(self, text: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        text = "hello"
        assert count_tokens(text) == self._tiktoken_count(text)

    def test_short_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == self._tiktoken_count(text)

    def test_longer_text(self):
        text = (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "as opposed to the natural intelligence displayed by animals and humans. "
            "AI research has been defined as the field of study of intelligent agents, "
            "which refers to any system that perceives its environment and takes actions "
            "that maximize its chance of achieving its goals."
        )
        assert count_tokens(text) == self._tiktoken_count(text)

    def test_code_snippet(self):
        text = "def hello_world():\n    print('Hello, world!')\n"
        assert count_tokens(text) == self._tiktoken_count(text)

    def test_return_type_is_int(self):
        assert isinstance(count_tokens("test"), int)

    def test_longer_text_nonzero(self):
        assert count_tokens("This is a test sentence.") > 0


# ---------------------------------------------------------------------------
# estimate_output_tokens
# ---------------------------------------------------------------------------

class TestEstimateOutputTokens:
    """Verify heuristic output token estimates."""

    EXPECTED = {
        "code_generation": 800,
        "code_review": 400,
        "summarization": 300,
        "tool_call": 150,
        "planning": 500,
        "chat": 200,
    }

    def test_all_task_types(self):
        for task, expected in self.EXPECTED.items():
            assert estimate_output_tokens(task) == expected, (
                f"estimate_output_tokens('{task}') expected {expected}"
            )

    def test_hyphen_variants_work(self):
        """Hyphenated task types are treated the same as underscored ones."""
        assert estimate_output_tokens("code-generation") == 800
        assert estimate_output_tokens("code-review") == 400
        assert estimate_output_tokens("tool-call") == 150

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            estimate_output_tokens("not_a_real_task")

    def test_return_type_is_int(self):
        assert isinstance(estimate_output_tokens("chat"), int)


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    """Verify model alias and fuzzy resolution."""

    def test_canonical_names_pass_through(self):
        for name in PRICING:
            assert resolve_model(name) == name

    def test_aliases_resolve(self):
        for alias, canonical in MODEL_ALIASES.items():
            assert resolve_model(alias) == canonical

    def test_fuzzy_partial_match(self):
        # "haiku-4-5" should fuzzy-match "claude-haiku-4-5"
        assert resolve_model("haiku-4-5") == "claude-haiku-4-5"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_model("totally-unknown-model-xyz")


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_list(self):
        assert isinstance(list_models(), list)

    def test_contains_required_models(self):
        models = list_models()
        for m in ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5", "gpt-4o", "gpt-4o-mini"]:
            assert m in models
