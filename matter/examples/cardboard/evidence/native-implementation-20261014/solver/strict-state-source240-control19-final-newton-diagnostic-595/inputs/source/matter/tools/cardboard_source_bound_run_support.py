"""Small testable helpers for source-bound subprocess timeout accounting."""

from __future__ import annotations

import signal
import subprocess
import json
from pathlib import Path
from typing import Any, Callable


def _record_signal_error(state: dict[str, Any], signum: int, error: OSError) -> None:
    state.setdefault("termination_errors", []).append({
        "signal": int(signum),
        "signal_name": signal.Signals(signum).name,
        "error_type": type(error).__name__,
        "reason": str(error),
        "errno": error.errno,
    })


def wait_with_timeout(
    process: Any,
    timeout_seconds: float,
    *,
    state: dict[str, Any],
    kill_process_group: Callable[[int, int], None],
    term_grace_seconds: float = 20.0,
) -> int:
    """Wait for a process and record timeout/termination failures faithfully.

    ``state`` is initialized by the caller before input sealing so the same
    timeout fact remains available if a signal or later wait raises. A failed
    SIGTERM is retained but does not prevent the bounded grace wait. A failed
    SIGKILL is retained and re-raised unchanged; the caller's exception
    receipt can therefore preserve the original OS exception and the timeout
    flag together.
    """
    state.setdefault("timed_out", False)
    state.setdefault("termination_errors", [])
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        state["timed_out"] = True

    try:
        kill_process_group(process.pid, signal.SIGTERM)
    except OSError as error:
        _record_signal_error(state, signal.SIGTERM, error)

    try:
        return process.wait(timeout=term_grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    try:
        kill_process_group(process.pid, signal.SIGKILL)
    except OSError as error:
        _record_signal_error(state, signal.SIGKILL, error)
        raise
    return process.wait()


def failure_receipt_fields(
    error: BaseException,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Return the stable error fields shared by runner exception receipts."""
    return {
        "exit_code": 1,
        "timed_out": state.get("timed_out") is True,
        "termination_errors": list(state.get("termination_errors", [])),
        "error_type": type(error).__name__,
        "reason": str(error),
    }


def write_failure_receipt(
    path: Path,
    error: BaseException,
    *,
    state: dict[str, Any],
    execution_started: bool,
    wall_seconds: float,
    finished_at_utc: str,
) -> dict[str, Any]:
    """Persist an exception receipt without relabeling timeout or OS failure."""
    receipt = failure_receipt_fields(error, state)
    receipt.update({
        "schema": "numi.cardboard.native-run-receipt.v2",
        "execution_started": execution_started,
        # An exception can interrupt the final source-hash scan; unknown is
        # distinct from either verified-unchanged or detected mutation.
        "source_inputs_unchanged": None,
        "changed_inputs": None,
        "stage": "execution" if execution_started else "input-seal",
        "wall_seconds": wall_seconds,
        "finished_at_utc": finished_at_utc,
    })
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
