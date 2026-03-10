"""
Pipeline Tracker — Structured Error Tracking & Run Logging
===========================================================

Provides:
  - PipelineTracker: captures step timing, success/failure, error context
  - @track_step decorator: wraps any pipeline function with auto-tracking
  - JSON run logs in outputs/pipeline_runs/
  - Optional DB logging to pipeline_runs table

Usage:
    from src.monitoring import PipelineTracker, track_step

    tracker = PipelineTracker("filter_v5")

    @track_step(tracker)
    def train_model(config):
        ...

    tracker.save_report()
"""

import functools
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class StepRecord:
    """Captures metadata for a single pipeline step."""

    __slots__ = (
        "name", "started_at", "ended_at", "duration_s",
        "status", "error", "error_trace", "inputs_summary",
        "outputs_summary", "metrics",
    )

    def __init__(self, name: str):
        self.name = name
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None
        self.duration_s: float = 0.0
        self.status: str = "running"
        self.error: Optional[str] = None
        self.error_trace: Optional[str] = None
        self.inputs_summary: dict = {}
        self.outputs_summary: dict = {}
        self.metrics: dict = {}

    def finish(self, status: str = "success"):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status

    def fail(self, exc: Exception):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = "failed"
        self.error = f"{type(exc).__name__}: {exc}"
        self.error_trace = traceback.format_exc()

    def to_dict(self) -> dict:
        return {
            k: getattr(self, k) for k in self.__slots__
            if getattr(self, k) is not None
        }


class PipelineTracker:
    """
    Tracks a pipeline run: steps, timing, errors, and summary.

    Args:
        run_name: Human-readable name (e.g. 'filter_v5_retrain')
        output_dir: Where to write JSON logs (default: outputs/pipeline_runs)
        notifier: Optional Notifier instance for failure alerts
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str = "outputs/pipeline_runs",
        notifier: Optional[Any] = None,
    ):
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier

        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: list[StepRecord] = []
        self._current_step: Optional[StepRecord] = None
        self._start_time = time.time()

        log.info(f"[TRACK] Pipeline run started: {run_name} [{self.run_id}]")

    # ── Step management ──────────────────────────────────────

    def begin_step(self, name: str, inputs: Optional[dict] = None) -> StepRecord:
        """Start tracking a new pipeline step."""
        step = StepRecord(name)
        if inputs:
            # Truncate large input summaries
            step.inputs_summary = {
                k: _truncate(v) for k, v in inputs.items()
            }
        self._current_step = step
        self.steps.append(step)
        log.info(f"  ▶ Step: {name}")
        return step

    def end_step(
        self,
        outputs: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ):
        """Mark the current step as completed successfully."""
        if not self._current_step:
            return

        step = self._current_step
        step.finish("success")
        step.duration_s = round(
            (datetime.fromisoformat(step.ended_at)
             - datetime.fromisoformat(step.started_at)).total_seconds(), 2
        )
        if outputs:
            step.outputs_summary = {k: _truncate(v) for k, v in outputs.items()}
        if metrics:
            step.metrics = metrics

        log.info(f"  [OK] Step complete: {step.name} ({step.duration_s}s)")
        self._current_step = None

    def fail_step(self, exc: Exception):
        """Mark the current step as failed with error context."""
        if not self._current_step:
            return

        step = self._current_step
        step.fail(exc)
        step.duration_s = round(
            (datetime.fromisoformat(step.ended_at)
             - datetime.fromisoformat(step.started_at)).total_seconds(), 2
        )

        log.error(f"  [FAIL] Step failed: {step.name} -- {step.error}")

        # Auto-notify on failure
        if self.notifier:
            try:
                self.notifier.send_error(
                    run_name=self.run_name,
                    step_name=step.name,
                    error=step.error,
                    trace=step.error_trace,
                )
            except Exception:
                log.warning("Failed to send error notification")

        self._current_step = None

    # ── Report ───────────────────────────────────────────────

    def summary(self) -> dict:
        """Generate a run summary."""
        total_time = round(time.time() - self._start_time, 2)
        n_ok = sum(1 for s in self.steps if s.status == "success")
        n_fail = sum(1 for s in self.steps if s.status == "failed")

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "total_seconds": total_time,
            "steps_total": len(self.steps),
            "steps_ok": n_ok,
            "steps_failed": n_fail,
            "status": "failed" if n_fail > 0 else "success",
            "steps": [s.to_dict() for s in self.steps],
        }

    def save_report(self) -> Path:
        """Write the run report as a JSON file."""
        report = self.summary()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{self.run_name}_{self.run_id}.json"
        path = self.output_dir / filename

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info(f"[REPORT] Run report saved: {path}")

        # Send completion notification
        if self.notifier:
            try:
                self.notifier.send_run_complete(report)
            except Exception:
                log.warning("Failed to send completion notification")

        return path

    def log_to_db(self, conn):
        """Log the run summary to the pipeline_runs table."""
        report = self.summary()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          SERIAL PRIMARY KEY,
                run_id      TEXT UNIQUE,
                run_name    TEXT,
                started_at  TIMESTAMPTZ,
                duration_s  FLOAT,
                status      TEXT,
                steps_total INT,
                steps_ok    INT,
                steps_failed INT,
                report_json JSONB,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            INSERT INTO pipeline_runs
                (run_id, run_name, started_at, duration_s, status,
                 steps_total, steps_ok, steps_failed, report_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING;
        """, (
            report["run_id"],
            report["run_name"],
            report["started_at"],
            report["total_seconds"],
            report["status"],
            report["steps_total"],
            report["steps_ok"],
            report["steps_failed"],
            json.dumps(report),
        ))
        conn.commit()
        log.info(f"[DB] Run logged to DB: {report['run_id']}")


# ── Decorator ────────────────────────────────────────────────

def track_step(
    tracker: PipelineTracker,
    name: Optional[str] = None,
):
    """
    Decorator that auto-tracks a function as a pipeline step.

    Usage:
        @track_step(tracker, name="Train LightGBM")
        def train_model(config):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        step_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Capture input summary (first 5 kwargs)
            inputs = {k: v for i, (k, v) in enumerate(kwargs.items()) if i < 5}
            tracker.begin_step(step_name, inputs=inputs)
            try:
                result = fn(*args, **kwargs)
                tracker.end_step(
                    outputs={"result_type": type(result).__name__}
                    if result is not None else None
                )
                return result
            except Exception as exc:
                tracker.fail_step(exc)
                raise

        return wrapper
    return decorator


# ── Helpers ──────────────────────────────────────────────────

def _truncate(v, max_len: int = 200) -> str:
    """Truncate a value to a short string for logging."""
    s = str(v)
    return s[:max_len] + "..." if len(s) > max_len else s
