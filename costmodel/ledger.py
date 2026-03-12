"""Cost ledger: SQLite persistence for all real API calls."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path.home() / ".ai-cost-modeler" / "ledger.db"

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
    called_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id TEXT PRIMARY KEY,
    architecture_name TEXT,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL,
    completed_at TEXT NOT NULL
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
    """SQLite-backed ledger for recording real API call costs."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def record_call(
        self,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        architecture_name: str = "",
        stage_name: str = "",
        cached_input_tokens: int = 0,
        latency_ms: int | None = None,
    ) -> int:
        """Insert one API call record. Returns the new row id."""
        now = datetime.utcnow().isoformat()
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
                architecture_name,
                stage_name,
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
    ) -> RunSummary:
        """Compute and store a run summary from all recorded calls for run_id."""
        row = self._conn.execute(
            """
            SELECT
                SUM(cost_usd)       AS total_cost,
                SUM(input_tokens + output_tokens) AS total_tokens
            FROM api_calls
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        total_cost = row["total_cost"] or 0.0
        total_tokens = row["total_tokens"] or 0
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_summaries
              (run_id, architecture_name, total_cost_usd, total_tokens,
               duration_seconds, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, architecture_name, total_cost, total_tokens, duration_seconds, now),
        )
        self._conn.commit()
        return RunSummary(
            run_id=run_id,
            architecture_name=architecture_name,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            completed_at=now,
        )

    def run_total(self, run_id: str) -> float:
        """Return total cost in USD for a given run_id."""
        row = self._conn.execute(
            "SELECT SUM(cost_usd) AS total FROM api_calls WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return float(row["total"] or 0.0)

    def recent_runs(
        self,
        limit: int = 50,
        architecture_name: str | None = None,
    ) -> list[RunSummary]:
        """Return the most recent run summaries."""
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
        return [
            RunSummary(
                run_id=r["run_id"],
                architecture_name=r["architecture_name"] or "",
                total_cost_usd=r["total_cost_usd"],
                total_tokens=r["total_tokens"],
                duration_seconds=r["duration_seconds"],
                completed_at=r["completed_at"],
            )
            for r in rows
        ]

    def architecture_stats(self, architecture_name: str) -> ArchStats | None:
        """Return aggregate stats for an architecture (requires ≥1 run)."""
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
        """Return the retry rate for an architecture as a fraction (0.0–1.0+).

        The retry rate is computed as:
            (total API calls - number of runs) / number of runs

        A value of 0.0 means every run succeeded on the first attempt.
        A value of 0.5 means runs needed 50% more calls than a no-retry baseline.

        Returns None if there are fewer than 5 runs (not enough data).
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

        # retry_rate = extra calls per run
        return (total_calls - run_count) / run_count

    def calls_for_model(
        self,
        model: str,
        limit: int = 200,
    ) -> list[ApiCallRecord]:
        """Return recent API calls for a specific model."""
        rows = self._conn.execute(
            """
            SELECT * FROM api_calls
            WHERE model LIKE ?
            ORDER BY called_at DESC
            LIMIT ?
            """,
            (f"%{model}%", limit),
        ).fetchall()
        return [
            ApiCallRecord(
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
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
