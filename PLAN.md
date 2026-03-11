# AI Inference Cost Modeler — Implementation Plan

## What We're Building

A tool that models and compares AI inference costs for different system architectures before you build them — and validates estimates against actual run data. Given a task and an architecture (single agent vs. multi-agent, model choices, caching strategies), compute expected cost per run and per 1000 runs.

## Why This Matters

The AI Breakdown episode on Anthropic's code review product ($15-25 per review) triggered a real debate: **usage-based AI pricing at scale is an unsolved product design problem**. Every team building on top of LLMs will face this reckoning.

Three problems this tool solves:
1. **Before building**: compare architectures on cost before writing a line of code
2. **While running**: track actual spend vs. estimates in real time
3. **After running**: understand where tokens went and what drove cost

The Boris Cherny insight — "use the best model, it's actually cheaper because it requires less correction" — is unverified intuition for most people. This tool lets you validate it empirically with real data from your own runs.

## Architecture

```
costmodel/
├── __init__.py
├── pricing.py       # Model price table, token counting utilities
├── models.py        # Architecture dataclasses: Pipeline, Stage, ModelCall
├── estimator.py     # Estimate cost from architecture description
├── ledger.py        # Record actual API calls and costs (SQLite)
├── comparator.py    # Side-by-side architecture comparison
├── reporter.py      # Report generator: estimate vs. actual breakdown
└── cli.py           # `cost estimate`, `cost compare`, `cost report`, `cost track`

integrations/
├── nano_agent.py    # Hook into nano-agent's run logger
└── anthropic_sdk.py # Middleware to intercept and record API calls

tests/
└── test_estimator.py

pyproject.toml
README.md
```

## Pricing Table (pricing.py)

Maintain a hardcoded but versioned price table. Source: Anthropic/OpenAI pricing pages.

```python
PRICING = {
    "claude-opus-4-6": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_write_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    },
    "claude-sonnet-4-6": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "cache_write_per_mtok": 3.75,
        "cache_read_per_mtok": 0.30,
    },
    "claude-haiku-4-5": {
        "input_per_mtok": 0.80,
        "output_per_mtok": 4.00,
        "cache_write_per_mtok": 1.00,
        "cache_read_per_mtok": 0.08,
    },
    "gpt-4o": { ... },
    "gpt-4o-mini": { ... },
}

def cost_for_call(model: str, input_tokens: int, output_tokens: int,
                  cached_input_tokens: int = 0) -> float:
    """Compute exact cost in USD for one API call."""
```

### Token estimation

```python
def estimate_tokens(text: str, model: str = "claude-sonnet-4-6") -> int:
    """Estimate token count using tiktoken approximation."""

def estimate_output_tokens(task_type: str) -> int:
    """Heuristic output token estimates by task type:
    code_generation: 800, code_review: 400, summarization: 300,
    tool_call: 150, planning: 500
    """
```

## Implementation Phases

### Phase 1: Architecture model (models.py)

Describe a system architecture as a composition of `Stage` objects:

```python
@dataclass
class ModelCall:
    model: str
    input_tokens: int           # estimated
    output_tokens: int          # estimated
    cached_input_tokens: int    # estimated cache hit tokens
    repeats: int = 1            # how many times this call happens per stage

@dataclass
class Stage:
    name: str
    calls: list[ModelCall]
    parallel: bool = False      # do calls run in parallel? (affects latency not cost)
    condition: float = 1.0      # probability this stage runs (0.0-1.0)

@dataclass
class Architecture:
    name: str
    stages: list[Stage]
    description: str = ""
```

Example: a 3-agent code review architecture:

```python
code_review_3agent = Architecture(
    name="3-agent parallel review",
    stages=[
        Stage("planning", [ModelCall("claude-opus-4-6", 2000, 500)]),
        Stage("review", [
            ModelCall("claude-sonnet-4-6", 8000, 1000, repeats=3)
        ], parallel=True),
        Stage("synthesis", [ModelCall("claude-opus-4-6", 5000, 800)]),
    ]
)
```

### Phase 2: Cost estimator (estimator.py)

Compute expected cost from an architecture description:

```python
@dataclass
class CostEstimate:
    architecture_name: str
    per_run_usd: float
    per_1000_runs_usd: float
    per_stage: dict[str, float]     # stage name → cost
    token_breakdown: dict           # input/output/cache per stage
    latency_estimate_seconds: float # sequential stages sum, parallel stages max
    confidence: str                 # "high" | "medium" | "low" (based on token estimate quality)

def estimate(arch: Architecture) -> CostEstimate:
    """Compute cost estimate for one run of this architecture."""
```

### Phase 3: Cost ledger (ledger.py)

Record every real API call that happens. Plugs into `nano-agent`'s existing run logger.

```sql
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    architecture_name TEXT,
    stage_name TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_input_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    called_at TIMESTAMP
);

CREATE TABLE run_summaries (
    run_id TEXT PRIMARY KEY,
    architecture_name TEXT,
    total_cost_usd REAL,
    total_tokens INTEGER,
    duration_seconds REAL,
    completed_at TIMESTAMP
);
```

```python
class CostLedger:
    def record_call(self, run_id, architecture, stage, model, usage) -> None
    def run_total(self, run_id) -> float
    def architecture_stats(self, architecture_name) -> ArchStats
        # avg cost, p50/p95 cost, avg tokens, run count
```

### Phase 4: Comparator (comparator.py)

Side-by-side comparison of multiple architectures, estimate vs. actual:

```python
comparator = ArchitectureComparator()
comparator.add(single_agent_haiku)
comparator.add(single_agent_opus)
comparator.add(three_agent_sonnet)
comparator.add(anthropic_code_review)  # $20 baseline

report = comparator.compare()
```

Output:
```
Architecture Comparison: code-review task
─────────────────────────────────────────────────────────────────────────
Architecture              Est/run   Act/run   At 1k/day    vs Anthropic
─────────────────────────────────────────────────────────────────────────
single-agent haiku        $0.004    $0.006    $6/day       333× cheaper
single-agent sonnet       $0.018    $0.021    $21/day      952× cheaper
single-agent opus         $0.045    $0.038    $38/day      526× cheaper
3-agent sonnet parallel   $0.054    $0.063    $63/day      317× cheaper
anthropic-code-review     $20.00    N/A       $20,000/day  baseline
─────────────────────────────────────────────────────────────────────────
Cherny hypothesis check:  opus ($0.038) vs haiku ($0.006)
  Opus is 6.3× more expensive per call
  If opus requires ≤ 1 retry per 5 haiku retries → opus wins on total cost
  Your actual retry rate: opus=0.08, haiku=0.34  → OPUS IS CHEAPER ✓
```

### Phase 5: Anthropic SDK middleware (integrations/anthropic_sdk.py)

Intercept all Anthropic API calls transparently to record real usage:

```python
from costmodel.integrations.anthropic_sdk import tracked_client

client = tracked_client(
    anthropic.Anthropic(),
    ledger=ledger,
    architecture="3-agent-review",
    stage="review"
)

# All calls through `client` are automatically recorded
response = client.messages.create(...)
```

Implemented by subclassing `anthropic.Anthropic` and overriding `messages.create()` to call `ledger.record_call()` before returning.

### Phase 6: Reporter (reporter.py)

Generate a cost analysis report after a set of runs:

```
Cost Report: nano-agent (last 50 runs)
Generated: 2026-03-11

Summary
  Total spent:          $0.87
  Avg per run:          $0.017
  Most expensive run:   $0.12 (task: "refactor the entire database module")
  Cheapest run:         $0.001 (task: "fix typo in README")

By Model
  claude-opus-4-6:      $0.61 (70%)  avg 12,400 tokens/call
  claude-sonnet-4-6:    $0.26 (30%)  avg 4,200 tokens/call

By Stage
  planning:             $0.12 (14%)
  tool_execution:       $0.58 (67%)
  synthesis:            $0.17 (19%)

Estimate Accuracy
  Avg error:            ±23% (estimate vs actual)
  Systematic bias:      +18% overestimate
  Worst miss:           +180% (long refactor task underestimated)

Token Distribution
  p50 per run:          820 tokens
  p95 per run:          8,400 tokens
  Max:                  42,000 tokens

Cherny Hypothesis Validation
  Model: claude-opus-4-6
  Retry rate:           8.2%
  Cost with retries:    $0.038/run effective
  vs haiku equivalent:  $0.024/run (haiku + 34% retry rate)
  Verdict:              ⚠ DEPENDS ON TASK TYPE
    Simple tasks (< 3 tool calls): haiku wins
    Complex tasks (> 5 tool calls): opus wins
```

### Phase 7: CLI

```bash
# Estimate before building
cost estimate --arch single-agent-opus --task-type code-review --runs 1000

# Compare architectures
cost compare arch1.yaml arch2.yaml --baseline "anthropic-code-review:$20"

# Track a live run (integrates with nano-agent)
cost track --arch my-agent --run-id abc123

# Generate report
cost report --arch my-agent --last 50 --output report.md

# Validate Cherny hypothesis on your actual data
cost cherny-check --model opus --vs haiku
```

### Phase 8: Validate Cherny hypothesis on real nano-agent runs

Run 20 tasks through `nano-agent` twice — once with Haiku, once with Opus. Record:
- Total cost (including retries/corrections)
- Number of attempts to complete task
- Success rate

Test the hypothesis: does Opus cost less in aggregate despite higher per-token price?

Write `CHERNY_VALIDATION.md` with real numbers.

## Key Design Decisions

**Why estimate from architecture descriptions, not just measure?**
Measurement requires running the system. Estimation lets you compare architectures before building. Both are necessary — the comparator shows estimate vs. actual to close the loop.

**Why a middleware approach for recording, not explicit logging?**
Explicit logging in every agent requires changing agent code. Middleware is transparent — the agent doesn't know it's being tracked. This is the right separation of concerns.

**Why validate the "Cherny hypothesis" explicitly?**
Because it's a testable, counterintuitive claim that most people don't believe without data. Providing real numbers either confirms or refutes it on your specific tasks. This is the most valuable output of the whole project.

**What we're NOT building**
- Budget alerts / hard spending limits (follow-on)
- Multi-provider routing optimizer (find cheapest model for each call type)
- Enterprise billing integration

## Acceptance Criteria

1. `cost estimate` produces a cost table matching manual calculation for a 3-stage architecture
2. SDK middleware records all API calls without changing agent behavior
3. `cost compare` shows Anthropic's $20/review correctly positioned vs. DIY alternatives
4. Cherny validation runs on ≥20 real task pairs, produces `CHERNY_VALIDATION.md` with real numbers
5. Estimate accuracy ≤ ±30% average error across 50 real runs

## Learning Outcomes

After building this you will understand:
- Real AI inference economics at different scales
- Why usage-based pricing creates a fundamentally different product design constraint than subscriptions
- Whether the "use the best model" advice holds for your actual task distribution
- How to model system costs before building (applies beyond AI to any metered service)
- Why p95 cost matters more than average cost for any real system
