"""Cost ledger: SQLite persistence for all real API calls."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path.home() / ".cost-modeler" / "ledger.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    architecture_name TEXT,
    stage_name TEXT,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    latency_ms INTEGER,
    called_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id TEXT PRIMARY KEY,
    architecture_name TEXT,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL,
    task TEXT,
    completed_at TIMESTAMP NOT NULL
);
"""


@dataclass
class ApiCallRecord:
    run_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    architecture_name: str = ""
    stage_name: str = ""
    cached_input_tokens: int = 0
    latency_ms: Optional[int] = None
    called_at: str = ""
    id: Optional[int] = None


@dataclass
class RunSummary:
    run_id: str
    architecture_name: str
    total_cost_usd: float
    total_tokens: int
    duration_seconds: Optional[float]
    completed_at: str
    task: Optional[str] = None


@dataclass
class ArchStats:
    architecture_name: str
    run_count: int
    avg_cost_usd: float
    p50_cost_usd: float
    p95_cost_usd: float
    avg_tokens: float
    total_cost_usd: float


class CostLedger:
    """SQLite-backed ledger for recording real API call costs.

    The database is auto-created at ``~/.cost-modeler/ledger.db`` (or the
    path supplied to ``__init__``).  All writes are committed immediately so
    the file is always consistent even if the process is killed.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_call(
        self,
        run_id: str,
        arch: str,
        stage: str,
        model: str,
        usage: dict,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
    ) -> int:
        """Insert one API call record and return the new row id.

        Parameters
        ----------
        run_id:
            Identifier for the logical run (group of calls).
        arch:
            Architecture name (e.g. ``"3-agent-sonnet"``).
        stage:
            Stage name within the architecture (e.g. ``"review"``).
        model:
            Model identifier (e.g. ``"claude-haiku-4-5"``).
        usage:
            Dict with keys ``input_tokens``, ``output_tokens``, and
            optionally ``cached_input_tokens``.  These map directly to the
            Anthropic / OpenAI usage objects.
        latency_ms:
            Wall-clock latency of the API call in milliseconds.
        cost_usd:
            Pre-computed cost.  If not supplied, the ledger will attempt to
            compute it from ``usage`` via :func:`costmodel.pricing.cost_usd`.
            If pricing is unavailable the value defaults to ``0.0``.
        """
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cached_input_tokens = int(usage.get("cached_input_tokens", 0))

        if cost_usd is None:
            try:
                from costmodel.pricing import cost_usd as _cost_usd

                cost_usd = _cost_usd(model, input_tokens, output_tokens, cached_input_tokens)
            except Exception:
                cost_usd = 0.0

        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO api_calls
              (run_id, architecture_name, stage_name, model,
               input_tokens, output_tokens, cached_input_tokens,
               cost_usd, latency_ms, called_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                arch,
                stage,
                model,
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cost_usd,
                latency_ms,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def finish_run(
        self,
        run_id: str,
        architecture_name: str = "",
        duration_seconds: float | None = None,
        task: str | None = None,
    ) -> RunSummary:
        """Compute and store a run summary from all recorded calls for *run_id*.

        Call this once after all :meth:`record_call` invocations for a run are
        done.  It upserts a row in ``run_summaries`` derived from the
        ``api_calls`` data.
        """
        row = self._conn.execute(
            """
            SELECT
                SUM(cost_usd)                       AS total_cost,
                SUM(input_tokens + output_tokens)   AS total_tokens
            FROM api_calls
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        total_cost = row["total_cost"] or 0.0
        total_tokens = row["total_tokens"] or 0
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_summaries
              (run_id, architecture_name, total_cost_usd, total_tokens,
               duration_seconds, task, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, architecture_name, total_cost, total_tokens, duration_seconds, task, now),
        )
        self._conn.commit()
        return RunSummary(
            run_id=run_id,
            architecture_name=architecture_name,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            task=task,
            completed_at=now,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def run_total(self, run_id: str) -> float:
        """Return the total cost in USD for all calls in *run_id*."""
        row = self._conn.execute(
            "SELECT SUM(cost_usd) AS total FROM api_calls WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return float(row["total"] or 0.0)

    def all_runs(self, last_n: int = 50) -> list[RunSummary]:
        """Return the *last_n* most recent run summaries (default 50).

        Rows are ordered newest-first by ``completed_at``.
        """
        rows = self._conn.execute(
            "SELECT * FROM run_summaries ORDER BY completed_at DESC LIMIT ?",
            (last_n,),
        ).fetchall()
        return [self._row_to_run_summary(r) for r in rows]

    def recent_runs(
        self,
        limit: int = 50,
        architecture_name: str | None = None,
    ) -> list[RunSummary]:
        """Return recent run summaries, optionally filtered by architecture.

        This is a convenience wrapper around :meth:`all_runs` that also
        supports filtering by ``architecture_name``.
        """
        if architecture_name:
            rows = self._conn.execute(
                """
                SELECT * FROM run_summaries
                WHERE architecture_name = ?
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                (architecture_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM run_summaries ORDER BY completed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_run_summary(r) for r in rows]

    def architecture_stats(self, architecture_name: str) -> ArchStats | None:
        """Return aggregate cost statistics for *architecture_name*.

        Returns ``None`` when no runs exist for the given architecture.

        The returned :class:`ArchStats` includes:

        * ``run_count`` — total number of completed runs
        * ``avg_cost_usd`` — arithmetic mean of per-run total cost
        * ``p50_cost_usd`` — median per-run cost
        * ``p95_cost_usd`` — 95th-percentile per-run cost
        * ``avg_tokens`` — average total tokens per run
        * ``total_cost_usd`` — sum of all run costs
        """
        rows = self._conn.execute(
            """
            SELECT total_cost_usd, total_tokens
            FROM run_summaries
            WHERE architecture_name = ?
            ORDER BY completed_at DESC
            """,
            (architecture_name,),
        ).fetchall()
        if not rows:
            return None
        costs = sorted([r["total_cost_usd"] for r in rows])
        tokens = [r["total_tokens"] for r in rows]
        n = len(costs)
        p50 = costs[int(n * 0.50)]
        p95 = costs[min(int(n * 0.95), n - 1)]
        return ArchStats(
            architecture_name=architecture_name,
            run_count=n,
            avg_cost_usd=sum(costs) / n,
            p50_cost_usd=p50,
            p95_cost_usd=p95,
            avg_tokens=sum(tokens) / n,
            total_cost_usd=sum(costs),
        )

    def retry_rate_for_architecture(self, architecture_name: str) -> Optional[float]:
        """Return the retry rate for *architecture_name* as a fraction.

        The retry rate is computed as::

            (total API calls - number of runs) / number of runs

        A value of ``0.0`` means every run succeeded on the first attempt.
        A value of ``0.5`` means runs needed 50% more calls than a no-retry
        baseline.

        Returns ``None`` when there are fewer than 5 runs (insufficient data).
        """
        row = self._conn.execute(
            """
            SELECT COUNT(DISTINCT run_id) AS run_count
            FROM api_calls
            WHERE architecture_name = ?
            """,
            (architecture_name,),
        ).fetchone()
        run_count = row["run_count"] if row else 0
        if run_count < 5:
            return None

        call_row = self._conn.execute(
            """
            SELECT COUNT(*) AS total_calls
            FROM api_calls
            WHERE architecture_name = ?
            """,
            (architecture_name,),
        ).fetchone()
        total_calls = call_row["total_calls"] if call_row else 0
        if run_count == 0:
            return None

        return (total_calls - run_count) / run_count

    def calls_for_model(
        self,
        model: str,
        limit: int = 200,
    ) -> list[ApiCallRecord]:
        """Return recent API calls for a specific *model* (substring match)."""
        rows = self._conn.execute(
            """
            SELECT * FROM api_calls
            WHERE model LIKE ?
            ORDER BY called_at DESC
            LIMIT ?
            """,
            (f"%{model}%", limit),
        ).fetchall()
        return [self._row_to_api_call(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "CostLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private converters
    # ------------------------------------------------------------------

    def _row_to_run_summary(self, r: sqlite3.Row) -> RunSummary:
        return RunSummary(
            run_id=r["run_id"],
            architecture_name=r["architecture_name"] or "",
            total_cost_usd=r["total_cost_usd"],
            total_tokens=r["total_tokens"],
            duration_seconds=r["duration_seconds"],
            task=r["task"] if "task" in r.keys() else None,
            completed_at=r["completed_at"],
        )

    def _row_to_api_call(self, r: sqlite3.Row) -> ApiCallRecord:
        return ApiCallRecord(
            id=r["id"],
            run_id=r["run_id"],
            architecture_name=r["architecture_name"] or "",
            stage_name=r["stage_name"] or "",
            model=r["model"],
            input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"],
            cached_input_tokens=r["cached_input_tokens"],
            cost_usd=r["cost_usd"],
            latency_ms=r["latency_ms"],
            called_at=r["called_at"],
        )
