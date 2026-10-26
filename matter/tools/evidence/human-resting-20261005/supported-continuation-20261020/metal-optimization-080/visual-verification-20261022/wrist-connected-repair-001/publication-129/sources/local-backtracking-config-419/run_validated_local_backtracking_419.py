"""Validate a 373 backtracking config before invoking its frozen runner."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OWNER_PATH = HERE / "local_backtracking_config_419.py"
OWNER_SHA256 = "ea14c34f3d02c8ab3a115dc83d4622f5d813d45dd7240362197b4f45964dfe91"
RUNNER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py")
RUNNER_SHA256 = "a91b410c979701a29ab32f376637db566abb745e066b629c000f6a37fea13212"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
MAX_TIMEOUT_SECONDS = 3600
GRADED_VALIDATOR_PATH = HERE / "graded_context_validation_419.py"
GRADED_VALIDATOR_SHA256 = "07bf66689449298aaf72a0c186d9b8294e98bc2535241e346092d91fd486a9de"
GRADED_HELPER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/graded-local-step-423/graded_local_step_423.py")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config_owner():
    if not OWNER_PATH.is_file() or sha256_file(OWNER_PATH) != OWNER_SHA256:
        raise ValueError("419 config validator source pin mismatch")
    spec = importlib.util.spec_from_file_location("local_backtracking_config_419_pinned", OWNER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned 419 config validator")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def load_graded_validator():
    if not GRADED_VALIDATOR_PATH.is_file() or sha256_file(GRADED_VALIDATOR_PATH) != GRADED_VALIDATOR_SHA256:
        raise ValueError("419 graded-context validator source pin mismatch")
    spec = importlib.util.spec_from_file_location("graded_context_validation_419_pinned", GRADED_VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned 419 graded-context validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_373(config_path: str | Path, *, timeout_seconds: int = 1800, run=subprocess.run) -> dict:
    """Validate all dispatch pins, then run the frozen 373 script without an outer slot."""
    config_path = Path(config_path).resolve()
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be an integer in [1,{MAX_TIMEOUT_SECONDS}]")
    if not config_path.is_file():
        raise ValueError("backtracking config file is missing")
    if not PYTHON.is_file():
        raise ValueError("pinned Python executable is missing")
    if not RUNNER_PATH.is_file() or sha256_file(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("373 runner source pin mismatch")
    config = json.loads(config_path.read_text())
    owner = load_config_owner()
    graded = config.get("helper") == str(GRADED_HELPER_PATH)
    child_env = None
    if graded:
        validator = load_graded_validator()
        binding = validator.validate_graded_context_419(config)
        child_env = os.environ.copy()
        child_env.update({
            "NUMI_GRADED_LOCAL_STEP_CONTEXT_423": config["local_step_context_npz"],
            "NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423": config["local_step_context_sha256"],
            "NUMI_GRADED_LOCAL_STEP_KAPPA_423": str(config["local_step_context_kappa"]),
        })
    else:
        binding = owner.validate_mode_helper_419(config)
    out_value = config.get("out")
    if not isinstance(out_value, str) or not out_value:
        raise ValueError("backtracking config must name a fresh output directory")
    out_path = Path(out_value)
    if out_path.exists():
        raise ValueError("backtracking output directory already exists")
    command = [str(PYTHON), str(RUNNER_PATH), str(config_path)]
    try:
        kwargs = {"check": False, "timeout": timeout_seconds}
        if child_env is not None:
            kwargs["env"] = child_env
        result = run(command, **kwargs)
        returncode = int(result.returncode)
        status = "RUNNER_EXITED" if returncode == 0 else "RUNNER_NONZERO_EXIT"
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode = 124
        status = "RUNNER_TIMEOUT"
        timed_out = True
    return {
        "status": status,
        "command": command,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_binding": binding,
        "graded_context_environment_injected": child_env is not None,
        "python_path": str(PYTHON),
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": RUNNER_SHA256,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        receipt = execute_373(args.config, timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        print(json.dumps({"status": "REJECTED_BEFORE_RUNNER", "error": str(exc),
                          "native_admission": False, "accepted_physical_steps": 0}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return receipt["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
