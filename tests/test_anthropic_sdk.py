"""Tests for costmodel.integrations.anthropic_sdk — tracked_client middleware.

Acceptance criteria from issue #8:
  - tracked_client wraps anthropic.Anthropic() transparently.
  - Non-streaming: ledger contains a row with correct input/output tokens and
    cost_usd matching manual calculation via costmodel.pricing.cost_usd.
  - Streaming: usage accumulated from stream events is recorded correctly.
  - Attributes not overridden by TrackedClient are forwarded to the real client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from costmodel.ledger import CostLedger
from costmodel.pricing import cost_usd
from costmodel.integrations.anthropic_sdk import (
    TrackedClient,
    TrackedMessagesClient,
    TrackedStream,
    tracked_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path: Path) -> CostLedger:
    """Fresh ledger backed by a temp file."""
    return CostLedger(db_path=tmp_path / "test.db")


def _make_usage(
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """Return a mock Usage object mimicking the Anthropic SDK type."""
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    u.cache_read_input_tokens = cache_read_input_tokens
    return u


def _make_message_response(
    model: str = "claude-haiku-4-5",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read_input_tokens: int = 0,
) -> MagicMock:
    """Return a mock non-streaming Message response."""
    msg = MagicMock()
    msg.model = model
    msg.usage = _make_usage(input_tokens, output_tokens, cache_read_input_tokens)
    # Ensure it does NOT look like a Stream
    msg.__class__ = MagicMock  # prevent isinstance(msg, Stream) from being True
    return msg


def _make_mock_client(response: Any = None) -> MagicMock:
    """Build a mock anthropic.Anthropic client whose messages.create returns *response*."""
    import anthropic as anthropic_mod

    mock = MagicMock(spec=anthropic_mod.Anthropic)
    mock.messages = MagicMock()
    mock.messages.create.return_value = response or _make_message_response()
    return mock


# ---------------------------------------------------------------------------
# tracked_client factory
# ---------------------------------------------------------------------------


class TestTrackedClientFactory:
    def test_returns_tracked_client_instance(self, ledger: CostLedger) -> None:
        mock = _make_mock_client()
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="r1",
            architecture="arch",
            stage="stage",
        )
        assert isinstance(client, TrackedClient)

    def test_messages_is_tracked_messages_client(self, ledger: CostLedger) -> None:
        mock = _make_mock_client()
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="r1",
            architecture="arch",
            stage="stage",
        )
        assert isinstance(client.messages, TrackedMessagesClient)

    def test_non_messages_attrs_forwarded(self, ledger: CostLedger) -> None:
        """Attributes other than `messages` proxy to the real client."""
        import anthropic as anthropic_mod

        mock = _make_mock_client()
        mock.api_key = "sk-test-123"
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="r1",
            architecture="arch",
            stage="stage",
        )
        assert client.api_key == "sk-test-123"


# ---------------------------------------------------------------------------
# Non-streaming: create() records correct usage
# ---------------------------------------------------------------------------


class TestNonStreamingCreate:
    def _call(
        self,
        ledger: CostLedger,
        *,
        model: str = "claude-haiku-4-5",
        input_tokens: int = 1000,
        output_tokens: int = 200,
        cache_read_input_tokens: int = 0,
        run_id: str = "test-run",
    ) -> Any:
        response = _make_message_response(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        mock = _make_mock_client(response=response)
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id=run_id,
            architecture="test-arch",
            stage="test-stage",
        )
        result = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
        )
        return result

    def test_returns_original_response(self, ledger: CostLedger) -> None:
        result = self._call(ledger)
        assert result is not None
        assert hasattr(result, "usage")

    def test_ledger_has_one_row(self, ledger: CostLedger) -> None:
        self._call(ledger, run_id="single-row")
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE run_id='single-row'"
        ).fetchone()[0]
        assert count == 1

    def test_correct_input_tokens(self, ledger: CostLedger) -> None:
        self._call(ledger, input_tokens=1234, run_id="tok-check")
        row = ledger._conn.execute(
            "SELECT input_tokens FROM api_calls WHERE run_id='tok-check'"
        ).fetchone()
        assert row[0] == 1234

    def test_correct_output_tokens(self, ledger: CostLedger) -> None:
        self._call(ledger, output_tokens=567, run_id="out-check")
        row = ledger._conn.execute(
            "SELECT output_tokens FROM api_calls WHERE run_id='out-check'"
        ).fetchone()
        assert row[0] == 567

    def test_correct_cached_input_tokens(self, ledger: CostLedger) -> None:
        self._call(ledger, cache_read_input_tokens=300, run_id="cache-check")
        row = ledger._conn.execute(
            "SELECT cached_input_tokens FROM api_calls WHERE run_id='cache-check'"
        ).fetchone()
        assert row[0] == 300

    def test_cost_usd_matches_manual_calculation(self, ledger: CostLedger) -> None:
        """cost_usd in ledger must match costmodel.pricing.cost_usd."""
        model = "claude-haiku-4-5"
        input_tokens = 2000
        output_tokens = 400
        self._call(
            ledger,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            run_id="cost-match",
        )
        row = ledger._conn.execute(
            "SELECT cost_usd FROM api_calls WHERE run_id='cost-match'"
        ).fetchone()
        expected = cost_usd(model, input_tokens, output_tokens)
        assert abs(row[0] - expected) < 1e-9, (
            f"Ledger cost {row[0]!r} ≠ manual {expected!r}"
        )

    def test_latency_ms_recorded(self, ledger: CostLedger) -> None:
        self._call(ledger, run_id="latency-check")
        row = ledger._conn.execute(
            "SELECT latency_ms FROM api_calls WHERE run_id='latency-check'"
        ).fetchone()
        assert row[0] is not None
        assert row[0] >= 0

    def test_architecture_and_stage_stored(self, ledger: CostLedger) -> None:
        response = _make_message_response()
        mock = _make_mock_client(response=response)
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="arch-stage-check",
            architecture="my-arch",
            stage="my-stage",
        )
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
        )
        row = ledger._conn.execute(
            "SELECT architecture_name, stage_name FROM api_calls WHERE run_id='arch-stage-check'"
        ).fetchone()
        assert row[0] == "my-arch"
        assert row[1] == "my-stage"

    def test_real_client_create_called_with_kwargs(self, ledger: CostLedger) -> None:
        """The underlying client.messages.create must be called with the same kwargs."""
        mock = _make_mock_client()
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="passthrough",
            architecture="a",
            stage="s",
        )
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.5,
        )
        mock.messages.create.assert_called_once_with(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
            temperature=0.5,
        )

    def test_multiple_calls_all_recorded(self, ledger: CostLedger) -> None:
        mock = _make_mock_client()
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="multi-calls",
            architecture="a",
            stage="s",
        )
        for _ in range(3):
            client.messages.create(
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=50,
            )
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE run_id='multi-calls'"
        ).fetchone()[0]
        assert count == 3

    def test_run_total_matches_sum_of_manual_costs(self, ledger: CostLedger) -> None:
        """run_total() must equal sum of per-call manual costs."""
        model = "claude-haiku-4-5"
        calls = [
            (500, 100),
            (1000, 200),
            (750, 150),
        ]
        mock = _make_mock_client()
        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="run-total-check",
            architecture="a",
            stage="s",
        )
        expected_total = 0.0
        for inp, out in calls:
            resp = _make_message_response(input_tokens=inp, output_tokens=out)
            mock.messages.create.return_value = resp
            client.messages.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=256,
            )
            expected_total += cost_usd(model, inp, out)

        actual = ledger.run_total("run-total-check")
        assert abs(actual - expected_total) < 1e-9, (
            f"run_total={actual!r}, expected={expected_total!r}"
        )


# ---------------------------------------------------------------------------
# Streaming: TrackedStream accumulates usage and records once
# ---------------------------------------------------------------------------


def _make_stream_events(
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> list[MagicMock]:
    """Build a minimal sequence of stream events matching the Anthropic format."""
    # message_start event
    start = MagicMock()
    start.type = "message_start"
    start_usage = MagicMock()
    start_usage.input_tokens = input_tokens
    start_usage.output_tokens = 0
    start_usage.cache_read_input_tokens = 0
    start.message = MagicMock()
    start.message.usage = start_usage

    # content_block_delta (no usage)
    delta1 = MagicMock()
    delta1.type = "content_block_delta"

    # message_delta event — carries output token count
    msg_delta = MagicMock()
    msg_delta.type = "message_delta"
    delta_usage = MagicMock()
    delta_usage.output_tokens = output_tokens
    msg_delta.usage = delta_usage

    # message_stop
    stop = MagicMock()
    stop.type = "message_stop"

    return [start, delta1, msg_delta, stop]


class _FakeStream:
    """Minimal stand-in for anthropic.Stream."""

    def __init__(self, events: list[MagicMock]) -> None:
        self._events = events
        self.response = MagicMock()

    def __iter__(self) -> Iterator[MagicMock]:
        yield from self._events

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def close(self) -> None:
        pass


class TestTrackedStream:
    def test_yields_all_events(self, ledger: CostLedger) -> None:
        events = _make_stream_events()
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-yield",
            architecture="a",
            stage="s",
            model="claude-haiku-4-5",
            start_time=0.0,
        )
        collected = list(ts)
        assert len(collected) == len(events)

    def test_ledger_row_recorded_after_iteration(self, ledger: CostLedger) -> None:
        events = _make_stream_events(input_tokens=800, output_tokens=160)
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-record",
            architecture="a",
            stage="s",
            model="claude-haiku-4-5",
            start_time=0.0,
        )
        list(ts)  # consume the stream
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE run_id='stream-record'"
        ).fetchone()[0]
        assert count == 1

    def test_accumulated_tokens_correct(self, ledger: CostLedger) -> None:
        events = _make_stream_events(input_tokens=600, output_tokens=120)
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-tokens",
            architecture="a",
            stage="s",
            model="claude-haiku-4-5",
            start_time=0.0,
        )
        list(ts)
        row = ledger._conn.execute(
            "SELECT input_tokens, output_tokens FROM api_calls WHERE run_id='stream-tokens'"
        ).fetchone()
        assert row[0] == 600
        assert row[1] == 120

    def test_cost_matches_manual_calculation(self, ledger: CostLedger) -> None:
        model = "claude-haiku-4-5"
        input_tokens, output_tokens = 1000, 250
        events = _make_stream_events(input_tokens=input_tokens, output_tokens=output_tokens)
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-cost",
            architecture="a",
            stage="s",
            model=model,
            start_time=0.0,
        )
        list(ts)
        row = ledger._conn.execute(
            "SELECT cost_usd FROM api_calls WHERE run_id='stream-cost'"
        ).fetchone()
        expected = cost_usd(model, input_tokens, output_tokens)
        assert abs(row[0] - expected) < 1e-9

    def test_record_called_only_once(self, ledger: CostLedger) -> None:
        """Consuming the stream multiple times (or calling close) must not double-record."""
        events = _make_stream_events()
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-once",
            architecture="a",
            stage="s",
            model="claude-haiku-4-5",
            start_time=0.0,
        )
        list(ts)
        ts.close()  # should NOT record a second time
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE run_id='stream-once'"
        ).fetchone()[0]
        assert count == 1

    def test_context_manager_records_on_exit(self, ledger: CostLedger) -> None:
        events = _make_stream_events(input_tokens=500, output_tokens=100)
        fake = _FakeStream(events)
        ts = TrackedStream(
            fake,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="stream-ctx",
            architecture="a",
            stage="s",
            model="claude-haiku-4-5",
            start_time=0.0,
        )
        with ts:
            for _ in ts:
                pass
        count = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls WHERE run_id='stream-ctx'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Integration: streaming path via TrackedMessagesClient
# ---------------------------------------------------------------------------


class TestStreamingViaTrackedMessagesClient:
    """Test the streaming integration path through TrackedMessagesClient.

    We test the streaming path by patching the _is_stream helper so that our
    _FakeStream objects are treated as streams by the middleware, without
    needing to subclass the actual anthropic.Stream (which requires an HTTP
    response object at construction time).
    """

    def test_streaming_response_wrapped_in_tracked_stream(
        self, ledger: CostLedger
    ) -> None:
        """When messages.create returns a stream, TrackedMessagesClient wraps it."""
        events = _make_stream_events()
        fake_stream = _FakeStream(events)
        mock = _make_mock_client(response=fake_stream)

        # Patch _is_stream in the sdk module so _FakeStream is treated as a stream
        with patch(
            "costmodel.integrations.anthropic_sdk._is_stream",
            side_effect=lambda r: isinstance(r, _FakeStream),
        ):
            client = tracked_client(
                mock,
                ledger=ledger,
                run_id="stream-wrap",
                architecture="a",
                stage="s",
            )
            result = client.messages.create(
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=50,
                stream=True,
            )
            assert isinstance(result, TrackedStream)

    def test_stream_consumption_records_to_ledger(self, ledger: CostLedger) -> None:
        """Consuming the TrackedStream records usage in the ledger."""
        events = _make_stream_events(input_tokens=700, output_tokens=140)
        fake_stream = _FakeStream(events)
        mock = _make_mock_client(response=fake_stream)

        with patch(
            "costmodel.integrations.anthropic_sdk._is_stream",
            side_effect=lambda r: isinstance(r, _FakeStream),
        ):
            client = tracked_client(
                mock,
                ledger=ledger,
                run_id="stream-consume",
                architecture="a",
                stage="s",
            )
            result = client.messages.create(
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=50,
                stream=True,
            )
            list(result)  # consume

        row = ledger._conn.execute(
            "SELECT input_tokens, output_tokens FROM api_calls WHERE run_id='stream-consume'"
        ).fetchone()
        assert row[0] == 700
        assert row[1] == 140


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERIA (Issue #8)
# Run a tracked call; ledger contains correct tokens and cost.
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """End-to-end scenario that satisfies the issue acceptance criteria."""

    def test_tracked_client_records_correct_row(self, ledger: CostLedger) -> None:
        model = "claude-haiku-4-5"
        input_tokens = 3000
        output_tokens = 600

        # Simulate a real API response
        response = _make_message_response(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        mock = _make_mock_client(response=response)

        client = tracked_client(
            mock,
            ledger=ledger,
            run_id="acceptance-run",
            architecture="single-agent-haiku",
            stage="execution",
        )

        returned = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=256,
        )

        # 1. Response is returned unchanged
        assert returned is response

        # 2. Exactly one row in the ledger
        rows = ledger._conn.execute(
            "SELECT * FROM api_calls WHERE run_id='acceptance-run'"
        ).fetchall()
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

        row = rows[0]

        # 3. Token counts are correct
        assert row["input_tokens"] == input_tokens
        assert row["output_tokens"] == output_tokens

        # 4. cost_usd matches manual calculation
        expected_cost = cost_usd(model, input_tokens, output_tokens)
        assert abs(row["cost_usd"] - expected_cost) < 1e-9, (
            f"cost_usd={row['cost_usd']!r} ≠ expected={expected_cost!r}"
        )

        # 5. Metadata stored correctly
        assert row["architecture_name"] == "single-agent-haiku"
        assert row["stage_name"] == "execution"
        assert row["model"] == model

        # 6. run_total() returns the same cost
        total = ledger.run_total("acceptance-run")
        assert abs(total - expected_cost) < 1e-9
