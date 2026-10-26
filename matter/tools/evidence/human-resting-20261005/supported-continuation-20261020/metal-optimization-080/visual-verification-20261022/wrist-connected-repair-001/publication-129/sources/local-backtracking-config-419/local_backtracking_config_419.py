"""Fail-closed configuration binding for local/global 373 backtracking modes.

This adapter validates configuration ownership; exact geometry gates remain in
run_local_witness_backtracking_373.py and its audit owners.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
H = ROOT / "barycentric-connected-clearance-170"
LOCAL_HELPER = ROOT / "D/local-witness-backtracking-183/final-a644/local_witness_backtracking.py"
LOCAL_SHA256 = "a644de24e2f23e90807eb75578309618d89c1a3f9d95d4f6a6f045f4d935c31a"
LOCAL_ENTRY = "reduce_witness_node_scales"
GLOBAL_HELPER = H / "global_step_backtracking_275.py"
GLOBAL_SHA256 = "18891b1f700592563e139edfc8bd1361d230ff6fd5af2ae5ee036c9625c1826a"
GLOBAL_ENTRY = "reduce_global_step_scales"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("backtracking_helper_419", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned backtracking helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_signature(function, *, local: bool) -> None:
    sig = inspect.signature(function)
    names = set(sig.parameters)
    required = {"faces", "step_scales" if local else "scales",
                "bad_face_rows" if local else "bad_faces",
                "self_pair_face_rows" if local else "self_pairs",
                "anchors", "zero_below"}
    if not required.issubset(names):
        raise ValueError(f"helper signature missing required parameters: {sorted(required - names)}")
    if any(sig.parameters[k].kind is inspect.Parameter.POSITIONAL_ONLY for k in required):
        raise ValueError("helper parameters must support keyword arguments")


def validate_mode_helper_419(config: dict) -> dict:
    """Verify mode, source path/hash, and callable contract for a 373 config."""
    if not isinstance(config, dict):
        raise ValueError("backtracking config must be a mapping")
    mode = config.get("uniform_failure_backtracking")
    if type(mode) is not bool:
        raise ValueError("uniform_failure_backtracking must be an explicit boolean")
    if mode:
        expected_path, expected_hash, expected_entry, local = GLOBAL_HELPER, GLOBAL_SHA256, GLOBAL_ENTRY, False
        mode_name = "uniform"
    else:
        expected_path, expected_hash, expected_entry, local = LOCAL_HELPER, LOCAL_SHA256, LOCAL_ENTRY, True
        mode_name = "local"
    configured_path = config.get("helper")
    if not isinstance(configured_path, str) or Path(configured_path) != expected_path:
        raise ValueError(f"{mode_name} mode requires helper path {expected_path}")
    if config.get("helper_sha256") != expected_hash:
        raise ValueError(f"{mode_name} mode helper_sha256 pin mismatch")
    if not expected_path.is_file():
        raise ValueError("pinned helper file is missing")
    actual_hash = sha256_file(expected_path)
    if actual_hash != expected_hash:
        raise ValueError(f"pinned helper bytes changed: {actual_hash}")
    entry = config.get("helper_entry")
    if entry != expected_entry:
        raise ValueError(f"{mode_name} mode requires helper_entry={expected_entry}")
    module = _load_helper(expected_path)
    function = getattr(module, expected_entry, None)
    if not callable(function):
        raise ValueError("pinned helper entry is not callable")
    _check_signature(function, local=local)
    return {
        "status": "PASS_BACKTRACKING_MODE_HELPER_BINDING",
        "mode": mode_name,
        "uniform_failure_backtracking": mode,
        "helper_path": str(expected_path),
        "helper_sha256": actual_hash,
        "helper_entry": expected_entry,
        "signature": str(inspect.signature(function)),
        "geometry_certified": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }


def configure_local_mode_419(config: dict) -> tuple[dict, dict]:
    """Return a copied config bound to the tested 183 local witness reducer."""
    if not isinstance(config, dict):
        raise ValueError("backtracking config must be a mapping")
    out = copy.deepcopy(config)
    out.update({
        "helper": str(LOCAL_HELPER),
        "helper_sha256": LOCAL_SHA256,
        "helper_entry": LOCAL_ENTRY,
        "uniform_failure_backtracking": False,
    })
    receipt = validate_mode_helper_419(out)
    receipt["configured_from_helper_path"] = config.get("helper")
    receipt["configured_from_helper_sha256"] = config.get("helper_sha256")
    receipt["configured_from_helper_entry"] = config.get("helper_entry")
    receipt["configuration_copied_without_mutating_input"] = True
    return out, receipt
