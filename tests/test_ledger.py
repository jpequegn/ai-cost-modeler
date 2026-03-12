"""Tests for costmodel.ledger — CostLedger SQLite persistence.

Acceptance criteria from issue #6:
  - Record 5 API calls across 2 runs.
  - run_total() returns the correct sum per run.
  - architecture_stats() shows correct avg cost over multiple runs.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from costmodel.ledger import ApiCallRecord, ArchStats, CostLedger, RunSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path: Path) -> CostLedger:
    """Return a fresh in-memory ledger backed by a temp file."""
    db = tmp_path / "test_ledger.db"
    return CostLedger(db_path=db)


# ---------------------------------------------------------------------------
# Schema & initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_db_file_created(self, tmp_path: Path) -> None:
        db = tmp_path / "sub" / "ledger.db"
        ledger = CostLedger(db_path=db)
        ledger.close()
        assert db.exists()

    def test_tables_exist(self, ledger: CostLedger) -> None:
        tables = {
            row[0]
            for row in ledger._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "api_calls" in tables
        assert "run_summaries" in tables

    def test_api_calls_columns(self, ledger: CostLedger) -> None:
        cols = {
            row[1]
            for row in ledger._conn.execute("PRAGMA table_info(api_calls)").fetchall()
        }
        expected = {
            "id", "run_id", "architecture_name", "stage_name",
            "model", "input_tokens", "output_tokens", "cached_input_tokens",
            "cost_usd", "latency_ms", "called_at",
        }
        assert expected.issubset(cols)

    def test_run_summaries_columns(self, ledger: CostLedger) -> None:
        cols = {
            row[1]
            for row in ledger._conn.execute("PRAGMA table_info(run_summaries)").fetchall()
        }
        expected = {
            "run_id", "architecture_name", "total_cost_usd", "total_tokens",
            "duration_seconds", "task", "completed_at",
        }
        assert expected.issubset(cols)

    def test_context_manager(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        with CostLedger(db_path=db) as ledger:
            ledger.record_call(
                "r1", "arch", "stage", "claude-haiku-4-5",
                {"input_tokens": 100, "output_tokens": 50},
            )
        # connection should be closed; accessing the file should still work
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# record_call
# ---------------------------------------------------------------------------


class TestRecordCall:
    def test_returns_row_id(self, ledger: CostLedger) -> None:
        row_id = ledger.record_call(
            "run-1", "arch-a", "stage-1", "claude-haiku-4-5",
            {"input_tokens": 1000, "output_tokens": 200},
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_ids_increment(self, ledger: CostLedger) -> None:
        id1 = ledger.record_call(
            "run-1", "arch-a", "s1", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50},
        )
        id2 = ledger.record_call(
            "run-1", "arch-a", "s2", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50},
        )
        assert id2 > id1

    def test_row_persisted_correctly(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-42", "my-arch", "my-stage", "gpt-4o-mini",
            {"input_tokens": 500, "output_tokens": 100, "cached_input_tokens": 200},
            latency_ms=123,
        )
        row = ledger._conn.execute(
            "SELECT * FROM api_calls WHERE run_id = 'run-42'"
        ).fetchone()
        assert row is not None
        assert row["run_id"] == "run-42"
        assert row["architecture_name"] == "my-arch"
        assert row["stage_name"] == "my-stage"
        assert row["model"] == "gpt-4o-mini"
        assert row["input_tokens"] == 500
        assert row["output_tokens"] == 100
        assert row["cached_input_tokens"] == 200
        assert row["latency_ms"] == 123
        assert row["called_at"] != ""

    def test_cost_auto_computed_from_pricing(self, ledger: CostLedger) -> None:
        """When cost_usd is not supplied it should be computed from pricing."""
        ledger.record_call(
            "run-cost", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 1000, "output_tokens": 200},
        )
        row = ledger._conn.execute(
            "SELECT cost_usd FROM api_calls WHERE run_id = 'run-cost'"
        ).fetchone()
        assert row["cost_usd"] > 0.0

    def test_cost_explicit_overrides_auto(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-ex", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 1000, "output_tokens": 200},
            cost_usd=0.9999,
        )
        row = ledger._conn.execute(
            "SELECT cost_usd FROM api_calls WHERE run_id = 'run-ex'"
        ).fetchone()
        assert abs(row["cost_usd"] - 0.9999) < 1e-9

    def test_cached_input_defaults_to_zero(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-c0", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50},
        )
        row = ledger._conn.execute(
            "SELECT cached_input_tokens FROM api_calls WHERE run_id = 'run-c0'"
        ).fetchone()
        assert row["cached_input_tokens"] == 0


# ---------------------------------------------------------------------------
# run_total
# ---------------------------------------------------------------------------


class TestRunTotal:
    def test_empty_run_returns_zero(self, ledger: CostLedger) -> None:
        assert ledger.run_total("nonexistent-run") == 0.0

    def test_single_call(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-single", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 1000, "output_tokens": 200},
            cost_usd=0.01,
        )
        assert abs(ledger.run_total("run-single") - 0.01) < 1e-9

    def test_multiple_calls_sum(self, ledger: CostLedger) -> None:
        for _ in range(3):
            ledger.record_call(
                "run-multi", "arch", "stage", "claude-haiku-4-5",
                {"input_tokens": 100, "output_tokens": 50},
                cost_usd=0.005,
            )
        assert abs(ledger.run_total("run-multi") - 0.015) < 1e-9

    def test_isolated_per_run(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-A", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.10,
        )
        ledger.record_call(
            "run-B", "arch", "stage", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.20,
        )
        assert abs(ledger.run_total("run-A") - 0.10) < 1e-9
        assert abs(ledger.run_total("run-B") - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# finish_run / all_runs
# ---------------------------------------------------------------------------


class TestFinishRun:
    def test_run_summary_created(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-fin", "my-arch", "s1", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
        )
        summary = ledger.finish_run(
            "run-fin", architecture_name="my-arch", duration_seconds=1.5, task="code_review"
        )
        assert isinstance(summary, RunSummary)
        assert summary.run_id == "run-fin"
        assert summary.architecture_name == "my-arch"
        assert abs(summary.total_cost_usd - 0.01) < 1e-9
        assert summary.total_tokens == 150
        assert summary.duration_seconds == 1.5
        assert summary.task == "code_review"

    def test_upsert_replaces_existing(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "run-ups", "arch", "s1", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
        )
        ledger.finish_run("run-ups", architecture_name="arch")
        # Add another call then re-finish
        ledger.record_call(
            "run-ups", "arch", "s2", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
        )
        summary2 = ledger.finish_run("run-ups", architecture_name="arch")
        assert abs(summary2.total_cost_usd - 0.02) < 1e-9


# ---------------------------------------------------------------------------
# all_runs
# ---------------------------------------------------------------------------


class TestAllRuns:
    def test_returns_list(self, ledger: CostLedger) -> None:
        result = ledger.all_runs()
        assert isinstance(result, list)

    def test_empty_when_no_runs(self, ledger: CostLedger) -> None:
        assert ledger.all_runs() == []

    def test_returns_run_summaries(self, ledger: CostLedger) -> None:
        for run_id in ("r1", "r2", "r3"):
            ledger.record_call(
                run_id, "arch", "s", "claude-haiku-4-5",
                {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
            )
            ledger.finish_run(run_id, architecture_name="arch")
        runs = ledger.all_runs()
        assert len(runs) == 3
        assert all(isinstance(r, RunSummary) for r in runs)

    def test_ordered_newest_first(self, ledger: CostLedger) -> None:
        for i in range(3):
            run_id = f"ord-{i}"
            ledger.record_call(
                run_id, "arch", "s", "claude-haiku-4-5",
                {"input_tokens": 100, "output_tokens": 50}, cost_usd=float(i),
            )
            ledger.finish_run(run_id, architecture_name="arch")
            # small sleep so timestamps differ
            time.sleep(0.01)
        runs = ledger.all_runs()
        assert runs[0].run_id == "ord-2"
        assert runs[-1].run_id == "ord-0"

    def test_last_n_limit(self, ledger: CostLedger) -> None:
        for i in range(10):
            run_id = f"lim-{i}"
            ledger.record_call(
                run_id, "arch", "s", "claude-haiku-4-5",
                {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
            )
            ledger.finish_run(run_id, architecture_name="arch")
        assert len(ledger.all_runs(last_n=5)) == 5


# ---------------------------------------------------------------------------
# architecture_stats
# ---------------------------------------------------------------------------


class TestArchitectureStats:
    def test_returns_none_for_unknown_arch(self, ledger: CostLedger) -> None:
        assert ledger.architecture_stats("unknown-arch") is None

    def test_single_run(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "r1", "single-arch", "s", "claude-haiku-4-5",
            {"input_tokens": 1000, "output_tokens": 200}, cost_usd=0.05,
        )
        ledger.finish_run("r1", architecture_name="single-arch")
        stats = ledger.architecture_stats("single-arch")
        assert stats is not None
        assert stats.run_count == 1
        assert abs(stats.avg_cost_usd - 0.05) < 1e-9
        assert abs(stats.p50_cost_usd - 0.05) < 1e-9
        assert abs(stats.p95_cost_usd - 0.05) < 1e-9

    def test_avg_over_multiple_runs(self, ledger: CostLedger) -> None:
        costs = [0.10, 0.20, 0.30]
        for i, c in enumerate(costs):
            run_id = f"avg-run-{i}"
            ledger.record_call(
                run_id, "avg-arch", "s", "claude-haiku-4-5",
                {"input_tokens": 1000, "output_tokens": 200}, cost_usd=c,
            )
            ledger.finish_run(run_id, architecture_name="avg-arch")
        stats = ledger.architecture_stats("avg-arch")
        assert stats is not None
        assert stats.run_count == 3
        assert abs(stats.avg_cost_usd - 0.20) < 1e-9  # (0.10+0.20+0.30)/3
        assert abs(stats.total_cost_usd - 0.60) < 1e-9
        assert abs(stats.p50_cost_usd - 0.20) < 1e-9  # median of sorted [0.10,0.20,0.30]

    def test_returns_arch_stats_type(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "r-type", "t-arch", "s", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=0.01,
        )
        ledger.finish_run("r-type", architecture_name="t-arch")
        stats = ledger.architecture_stats("t-arch")
        assert isinstance(stats, ArchStats)
        assert stats.architecture_name == "t-arch"

    def test_isolated_from_other_archs(self, ledger: CostLedger) -> None:
        ledger.record_call(
            "r-x", "arch-x", "s", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=1.00,
        )
        ledger.finish_run("r-x", architecture_name="arch-x")
        ledger.record_call(
            "r-y", "arch-y", "s", "claude-haiku-4-5",
            {"input_tokens": 100, "output_tokens": 50}, cost_usd=2.00,
        )
        ledger.finish_run("r-y", architecture_name="arch-y")
        stats_x = ledger.architecture_stats("arch-x")
        stats_y = ledger.architecture_stats("arch-y")
        assert abs(stats_x.avg_cost_usd - 1.00) < 1e-9  # type: ignore[union-attr]
        assert abs(stats_y.avg_cost_usd - 2.00) < 1e-9  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERIA (Issue #6)
# Record 5 API calls across 2 runs. run_total() correct.
# architecture_stats() shows correct avg over multiple runs.
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """End-to-end scenario that satisfies the issue acceptance criteria."""

    def test_five_calls_two_runs_run_total_and_arch_stats(
        self, ledger: CostLedger
    ) -> None:
        # ── Run 1: 3 calls ─────────────────────────────────────────────────
        # planning call
        ledger.record_call(
            "run-2026-001", "3-agent-sonnet", "planning",
            "claude-opus-4-6",
            {"input_tokens": 2000, "output_tokens": 500},
            latency_ms=450,
            cost_usd=0.0675,
        )
        # two parallel review calls
        for i in range(2):
            ledger.record_call(
                "run-2026-001", "3-agent-sonnet", "review",
                "claude-sonnet-4-6",
                {"input_tokens": 8000, "output_tokens": 1000, "cached_input_tokens": 1000},
                latency_ms=800 + i * 50,
                cost_usd=0.0390,
            )
        ledger.finish_run(
            "run-2026-001",
            architecture_name="3-agent-sonnet",
            duration_seconds=1.8,
            task="code_review",
        )

        # ── Run 2: 2 calls ─────────────────────────────────────────────────
        ledger.record_call(
            "run-2026-002", "3-agent-sonnet", "planning",
            "claude-opus-4-6",
            {"input_tokens": 2000, "output_tokens": 500},
            latency_ms=420,
            cost_usd=0.0675,
        )
        ledger.record_call(
            "run-2026-002", "3-agent-sonnet", "review",
            "claude-sonnet-4-6",
            {"input_tokens": 8000, "output_tokens": 1000},
            latency_ms=750,
            cost_usd=0.0390,
        )
        ledger.finish_run(
            "run-2026-002",
            architecture_name="3-agent-sonnet",
            duration_seconds=1.2,
            task="code_review",
        )

        # ── Verify total call count ────────────────────────────────────────
        total_rows = ledger._conn.execute(
            "SELECT COUNT(*) FROM api_calls"
        ).fetchone()[0]
        assert total_rows == 5, f"Expected 5 api_calls rows, got {total_rows}"

        # ── run_total() accuracy ───────────────────────────────────────────
        # run-1: 0.0675 + 0.0390 + 0.0390 = 0.1455
        run1_total = ledger.run_total("run-2026-001")
        assert abs(run1_total - 0.1455) < 1e-6, (
            f"run_total('run-2026-001') = {run1_total}, expected 0.1455"
        )

        # run-2: 0.0675 + 0.0390 = 0.1065
        run2_total = ledger.run_total("run-2026-002")
        assert abs(run2_total - 0.1065) < 1e-6, (
            f"run_total('run-2026-002') = {run2_total}, expected 0.1065"
        )

        # ── architecture_stats() accuracy ─────────────────────────────────
        stats = ledger.architecture_stats("3-agent-sonnet")
        assert stats is not None
        assert stats.run_count == 2

        # avg = (0.1455 + 0.1065) / 2 = 0.1260
        expected_avg = (0.1455 + 0.1065) / 2
        assert abs(stats.avg_cost_usd - expected_avg) < 1e-6, (
            f"avg_cost_usd = {stats.avg_cost_usd}, expected {expected_avg}"
        )

        # p50 = costs[int(2*0.50)] = costs[1] = 0.1455 (sorted [0.1065, 0.1455])
        assert abs(stats.p50_cost_usd - 0.1455) < 1e-6

        # p95 = costs[min(int(2*0.95), 1)] = costs[1] = 0.1455
        assert abs(stats.p95_cost_usd - 0.1455) < 1e-6

        # ── all_runs() returns both ────────────────────────────────────────
        all_runs = ledger.all_runs()
        run_ids = {r.run_id for r in all_runs}
        assert "run-2026-001" in run_ids
        assert "run-2026-002" in run_ids
