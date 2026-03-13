"""AI Cost Modeler — model and compare AI inference costs."""

__version__ = "0.1.0"

from costmodel.comparator import (
    ArchitectureComparator,
    ArchRow,
    ChernyCheckResult,
    ComparisonReport,
)
from costmodel.models import Architecture, Stage, ModelCall, BUILTIN_ARCHITECTURES
from costmodel.estimator import CostEstimate, estimate
from costmodel.ledger import CostLedger, ArchStats, RunSummary, ApiCallRecord
from costmodel.pricing import (
    PRICING,
    PRICING_VERSION,
    cost_usd,
    count_tokens,
    estimate_output_tokens,
)
from costmodel.architectures.presets import (
    SINGLE_AGENT_HAIKU,
    SINGLE_AGENT_SONNET,
    SINGLE_AGENT_OPUS,
    THREE_AGENT_SONNET,
    ANTHROPIC_CODE_REVIEW,
)

__all__ = [
    "ArchitectureComparator",
    "ArchRow",
    "ChernyCheckResult",
    "ComparisonReport",
    "Architecture",
    "Stage",
    "ModelCall",
    "BUILTIN_ARCHITECTURES",
    "CostEstimate",
    "estimate",
    "CostLedger",
    "ArchStats",
    "RunSummary",
    "ApiCallRecord",
    # pricing
    "PRICING",
    "PRICING_VERSION",
    "cost_usd",
    "count_tokens",
    "estimate_output_tokens",
    # architecture presets
    "SINGLE_AGENT_HAIKU",
    "SINGLE_AGENT_SONNET",
    "SINGLE_AGENT_OPUS",
    "THREE_AGENT_SONNET",
    "ANTHROPIC_CODE_REVIEW",
]
