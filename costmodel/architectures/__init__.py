"""Pre-built architecture presets for common AI pipeline patterns."""

from costmodel.architectures.presets import (
    SINGLE_AGENT_HAIKU,
    SINGLE_AGENT_SONNET,
    SINGLE_AGENT_OPUS,
    THREE_AGENT_SONNET,
    ANTHROPIC_CODE_REVIEW,
)

__all__ = [
    "SINGLE_AGENT_HAIKU",
    "SINGLE_AGENT_SONNET",
    "SINGLE_AGENT_OPUS",
    "THREE_AGENT_SONNET",
    "ANTHROPIC_CODE_REVIEW",
]
