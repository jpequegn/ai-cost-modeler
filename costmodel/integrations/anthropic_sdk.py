"""Transparent SDK middleware for the Anthropic Python client.

Drop-in replacement that intercepts every ``messages.create()`` call,
records usage to a :class:`~costmodel.ledger.CostLedger`, and returns the
original response unchanged.

Usage::

    import anthropic
    from costmodel.ledger import CostLedger
    from costmodel.integrations.anthropic_sdk import tracked_client

    ledger = CostLedger()
    client = tracked_client(
        anthropic.Anthropic(),
        ledger=ledger,
        run_id="abc123",
        architecture="single-agent-opus",
        stage="execution",
    )

    # All calls through client are recorded automatically
    response = client.messages.create(model=..., messages=..., max_tokens=...)
    # → ledger.record_call() called with real usage from response
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import anthropic
from anthropic import Stream
from anthropic.types import RawMessageStreamEvent


def _is_stream(response: Any) -> bool:
    """Return True if *response* is an Anthropic streaming response.

    Extracted into a standalone function so tests can patch it without
    fighting with ``unittest.mock``'s restrictions on ``__instancecheck__``.
    """
    return isinstance(response, Stream)


class TrackedStream:
    """Wraps a :class:`~anthropic.Stream` to record usage after iteration.

    Behaves identically to the original stream from the caller's perspective.
    Usage is recorded once the stream is fully consumed (or closed) via the
    ``message_delta`` event's accumulated token counts.

    The stream yields :class:`~anthropic.types.RawMessageStreamEvent` objects.
    We intercept ``message_start`` (which carries the initial usage snapshot)
    and ``message_delta`` events (which carry incremental output token counts)
    to accumulate the final usage, then call ``ledger.record_call()`` in the
    ``close``/``__exit__`` hook.
    """

    def __init__(
        self,
        raw_stream: Stream[RawMessageStreamEvent],
        *,
        ledger: Any,
        run_id: str,
        architecture: str,
        stage: str,
        model: str,
        start_time: float,
    ) -> None:
        self._raw_stream = raw_stream
        self._ledger = ledger
        self._run_id = run_id
        self._architecture = architecture
        self._stage = stage
        self._model = model
        self._start_time = start_time

        # Accumulated usage
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cached_input_tokens: int = 0
        self._recorded: bool = False

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[RawMessageStreamEvent]:
        try:
            for event in self._raw_stream:
                self._handle_event(event)
                yield event
        finally:
            self._record()

    def __enter__(self) -> "TrackedStream":
        self._raw_stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._raw_stream.__exit__(*args)
        self._record()

    def close(self) -> None:
        """Close the underlying stream and flush any pending ledger record."""
        self._raw_stream.close()
        self._record()

    # Expose the response attribute so callers can inspect headers/status
    @property
    def response(self) -> Any:
        return self._raw_stream.response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_event(self, event: RawMessageStreamEvent) -> None:
        """Extract usage counters from stream events."""
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            # ``message_start`` carries the initial Usage object on event.message
            msg = getattr(event, "message", None)
            if msg is not None:
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    self._input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                    self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                    self._cached_input_tokens += int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )

        elif event_type == "message_delta":
            # ``message_delta`` carries incremental output tokens in event.usage
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def _record(self) -> None:
        """Record the accumulated usage to the ledger (once only)."""
        if self._recorded:
            return
        self._recorded = True
        latency_ms = int((time.monotonic() - self._start_time) * 1000)
        self._ledger.record_call(
            run_id=self._run_id,
            arch=self._architecture,
            stage=self._stage,
            model=self._model,
            usage={
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "cached_input_tokens": self._cached_input_tokens,
            },
            latency_ms=latency_ms,
        )


class TrackedMessagesClient:
    """Proxy for ``anthropic.Anthropic().messages`` that records every call.

    Wraps the real ``Messages`` resource and overrides ``create()`` to:

    1. Delegate to the real client.
    2. Extract ``response.usage`` (for non-streaming calls) or wrap the stream
       (for streaming calls).
    3. Call :meth:`~costmodel.ledger.CostLedger.record_call` with the real
       token counts.
    4. Return the original response unchanged.

    All other attributes (``with_raw_response``, ``with_streaming_response``,
    ``batches``, etc.) are forwarded to the underlying resource via
    ``__getattr__``.
    """

    def __init__(
        self,
        messages_resource: Any,
        *,
        ledger: Any,
        run_id: str,
        architecture: str,
        stage: str,
    ) -> None:
        self._messages = messages_resource
        self._ledger = ledger
        self._run_id = run_id
        self._architecture = architecture
        self._stage = stage

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def create(self, **kwargs: Any) -> Any:
        """Call the real ``messages.create()`` and record usage.

        Returns the original response (non-streaming) or a
        :class:`TrackedStream` (streaming).
        """
        model: str = kwargs.get("model", "")
        start_time = time.monotonic()
        response = self._messages.create(**kwargs)
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Streaming path — wrap in TrackedStream so usage is recorded after
        # the caller consumes the iterator.
        if _is_stream(response):
            return TrackedStream(
                response,
                ledger=self._ledger,
                run_id=self._run_id,
                architecture=self._architecture,
                stage=self._stage,
                model=model,
                start_time=start_time,
            )

        # Non-streaming path — usage is available immediately.
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._ledger.record_call(
                run_id=self._run_id,
                arch=self._architecture,
                stage=self._stage,
                model=model,
                usage={
                    "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                    "cached_input_tokens": int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    ),
                },
                latency_ms=latency_ms,
            )
        return response

    # ------------------------------------------------------------------
    # Transparent proxying of all other attributes
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class TrackedClient:
    """Thin wrapper around :class:`anthropic.Anthropic` with cost tracking.

    Delegates every attribute access to the real client **except**
    ``messages``, which is replaced with a :class:`TrackedMessagesClient`.

    Construct via :func:`tracked_client` rather than instantiating directly.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        ledger: Any,
        run_id: str,
        architecture: str,
        stage: str,
    ) -> None:
        self._client = client
        self.messages = TrackedMessagesClient(
            client.messages,
            ledger=ledger,
            run_id=run_id,
            architecture=architecture,
            stage=stage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def tracked_client(
    client: anthropic.Anthropic,
    *,
    ledger: Any,
    run_id: str,
    architecture: str,
    stage: str,
) -> TrackedClient:
    """Return a :class:`TrackedClient` that wraps *client* with cost tracking.

    Every call made through the returned client's ``messages.create()`` is
    transparently forwarded to the real Anthropic API and recorded in *ledger*.

    Args:
        client: A real :class:`anthropic.Anthropic` instance.
        ledger: A :class:`~costmodel.ledger.CostLedger` instance (or any
            object with a compatible ``record_call()`` method).
        run_id: Logical run identifier, e.g. ``"abc123"``.
        architecture: Architecture label, e.g. ``"single-agent-opus"``.
        stage: Stage label within the architecture, e.g. ``"execution"``.

    Returns:
        A :class:`TrackedClient` drop-in replacement.

    Example::

        import anthropic
        from costmodel.ledger import CostLedger
        from costmodel.integrations.anthropic_sdk import tracked_client

        ledger = CostLedger()
        client = tracked_client(
            anthropic.Anthropic(),
            ledger=ledger,
            run_id="run-001",
            architecture="single-agent-haiku",
            stage="generation",
        )
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(ledger.run_total("run-001"))
    """
    return TrackedClient(
        client,
        ledger=ledger,
        run_id=run_id,
        architecture=architecture,
        stage=stage,
    )
