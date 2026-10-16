"""Strict, source-bound effective native environment contracts for cardboard runs.

Only source-registered native controls and two timing controls are modeled.
Inherited NM_FEM_* and NM_CONTACT_* variables are always scrubbed before launch; qualification
contracts preserve absent, empty, and non-empty values, while output-path
captures require a separate diagnostic-only contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

CONTRACT_SCHEMA = "numi.cardboard.effective-native-controls.v1"
DIAGNOSTIC_CONTRACT_SCHEMA = "numi.cardboard.effective-native-diagnostic-controls.v1"
RUN_SCHEMA = "numi.cardboard.native-environment-run.v1"
DIAGNOSTIC_RUN_SCHEMA = "numi.cardboard.native-diagnostic-environment-run.v1"
QUALIFIED_PRODUCERS = frozenset({"source-bound-native-runner", "resolution-study"})
DIAGNOSTIC_PRODUCER = "source-bound-diagnostic-runner"
_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Extracted from the combined source169 runtime and source167 probe
# environment read sites. The helper itself is included in study/source
# bindings; tests compare this registry to the candidate source so a new
# control cannot disappear.
NATIVE_ENV_KEYS = tuple(sorted({
    "NM_CONTACT_NODE_INCIDENCE_SORT",
    "NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR",
    "NM_FEM_BASE_CGLS_CAPTURE_NEWTON",
    "NM_FEM_BASE_CGLS_CAPTURE_PATH",
    "NM_FEM_BASE_CGLS_DAMPING_KG",
    "NM_FEM_BASE_CGLS_ITERATIONS",
    "NM_FEM_BASE_CGLS_PRIMARY_AFTER",
    "NM_FEM_BASE_CGLS_RIGHT_SCALE",
    "NM_FEM_BASE_CGLS_STATE_CAPTURE",
    "NM_FEM_BASE_CGLS_WARM_START_REFINE",
    "NM_FEM_CACHE_PROJECTION_REUSE",
    "NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_CONTACT_TRACE_ITERATION",
    "NM_FEM_CONTACT_TRACE_ROOT",
    "NM_FEM_COUPLED_CAPTURE_PATH",
    "NM_FEM_COUPLED_CAPTURE_SELECTOR",
    "NM_FEM_COUPLED_TRIAL_CAPTURE_PATH",
    "NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR",
    "NM_FEM_COUPLED_TRIAL_SUMMARY_PATH",
    "NM_FEM_COUPLED_TRIAL_SUMMARY_SELECTOR",
    "NM_FEM_COUPLED_TRIAL_VTCCD_AUDIT_SELECTOR",
    "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",
    "NM_FEM_FORCE_COMPLETED_TANGENT_CACHE_REBUILD",
    "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_FULL_CONTACT_REUSE_COHESIVE_SCRATCH",
    "NM_FEM_FULL_CONTACT_REUSE_SELF_SURFACE_SCRATCH",
    "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY",
    "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY",
    "NM_FEM_FULL_CONTACT_TILED_COHESIVE",
    "NM_FEM_FULL_CONTACT_TILED_COHESIVE_PAGE_FACES",
    "NM_FEM_GRAPH_PARALLEL_LU_SEARCH",
    "NM_FEM_NEWTON_TRACE_ROOT",
    "NM_FEM_OPERATOR_FD_CAPTURE_PATH",
    "NM_FEM_OPERATOR_FD_EPSILON_SCALE",
    "NM_FEM_OPERATOR_FD_SELECTOR",
    "NM_FEM_REGIONAL_ACCEPTED_CORRECTION_RECYCLE",
    "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH",
    "NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR",
    "NM_FEM_REGIONAL_CURRENT_RESTART_AUGMENTATION",
    "NM_FEM_REGIONAL_RAP_CAPTURE_ONLY",
    "NM_FEM_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
    "NM_FEM_STATUS5_CAPTURE_PATH",
    "NM_FEM_STATUS5_CAPTURE_ROOT",
    "NM_FEM_STRICT_INNER_FORCING",
    "NM_FEM_TRUE_RESIDUAL_CERTIFY",
    "NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR",
    "NM_FEM_TRUE_RESIDUAL_RESTART",
    "NM_FEM_TRUE_RESIDUAL_ROOT",
    "NUMI_CARDBOARD_STEP_TIMING",
    "NUMI_MATTER_GPU_TIMING_DENSE45",
}))

# Only these source controls are output paths. They can be requested only via
# --diagnostic-capture; the runner materializes a fresh case-local prefix.
CAPTURE_PATH_KEYS = frozenset(key for key in NATIVE_ENV_KEYS if key.endswith("_PATH"))
DIAGNOSTIC_CAPTURE_PREFIX_NAMES = {
    key: "capture-" + key.removeprefix("NM_FEM_").removesuffix("_PATH").lower().replace("_", "-")
    for key in CAPTURE_PATH_KEYS
}
MAX_DIAGNOSTIC_CAPTURE_FILES = 64
MAX_DIAGNOSTIC_CAPTURE_FILE_BYTES = 64 * 1024 * 1024
MAX_DIAGNOSTIC_CAPTURE_TOTAL_BYTES = 128 * 1024 * 1024
UINT32_MAX = (1 << 32) - 1
# These `_ROOT` controls select a control-step index; they are not filesystem
# roots. The Armijo roots additionally accept the source-defined `all` token.
NUMERIC_SELECTOR_KEYS = frozenset(key for key in NATIVE_ENV_KEYS if key.endswith("_ROOT"))
ALL_CONTROL_SELECTOR_KEYS = frozenset({
    "NM_FEM_TRUE_RESIDUAL_ROOT",
    "NM_FEM_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_CAPSULE_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_CAPSULE_BOX_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT",
    "NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT",
})
TIMING_KEYS = frozenset({"NUMI_CARDBOARD_STEP_TIMING", "NUMI_MATTER_GPU_TIMING_DENSE45"})
_SOURCE_KEY_RE = re.compile(r"\b(?:NM_FEM_[A-Z0-9_]+|NUMI_CARDBOARD_STEP_TIMING|NUMI_MATTER_GPU_TIMING_DENSE45)\b")
_CONTACT_READ_KEY_RE = re.compile(r'getenv\s*\(\s*"(NM_CONTACT_[A-Z0-9_]+)"')


def source_key_registry(source_root: os.PathLike[str] | str) -> tuple[str, ...]:
    root = Path(source_root)
    paths = (root / "matter/src/runtime.mm", root / "matter/tools/cardboard_probe.mm")
    found: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"native environment registry source is missing: {path}")
        text = path.read_text(encoding="utf-8")
        found.update(_SOURCE_KEY_RE.findall(text))
        found.update(_CONTACT_READ_KEY_RE.findall(text))
    return tuple(sorted(found))


def validate_source_registry(source_root: os.PathLike[str] | str) -> None:
    actual = source_key_registry(source_root)
    if actual != NATIVE_ENV_KEYS:
        missing = sorted(set(actual) - set(NATIVE_ENV_KEYS))
        extra = sorted(set(NATIVE_ENV_KEYS) - set(actual))
        raise ValueError(f"source native environment registry mismatch; unlisted={missing}, stale={extra}")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _contract_payload(contract: Mapping[str, object]) -> dict[str, object]:
    return {"schema": contract.get("schema"), "keys": contract.get("keys"),
            "effective": contract.get("effective")}


def _contract_digest(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(_contract_payload(contract))).hexdigest()


def _validate_environment_value(key: str, value: str) -> None:
    if "\x00" in value:
        raise ValueError(f"NUL is not allowed in environment value for {key}")
    if key == "NM_CONTACT_NODE_INCIDENCE_SORT" and value != "1":
        raise ValueError("NM_CONTACT_NODE_INCIDENCE_SORT must be exactly 1 when set")
    if key in NUMERIC_SELECTOR_KEYS:
        # Empty is a source-supported explicit disable for selector controls;
        # it remains distinct from an absent key in the hashed contract.
        if value == "":
            return
        if value == "all" and key in ALL_CONTROL_SELECTOR_KEYS:
            return
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"{key} is a numeric control-step selector, not a filesystem path")
        if int(value, 10) > UINT32_MAX:
            raise ValueError(f"{key} exceeds the native uint32 control-step range")


def parse_overrides(items: Sequence[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE arguments without collapsing duplicates/empty values."""
    overrides: dict[str, str] = {}
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("environment override must have KEY=VALUE form")
        key, value = item.split("=", 1)
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"malformed native environment key: {key!r}")
        if key not in NATIVE_ENV_KEYS:
            raise ValueError(f"unknown native environment key: {key}")
        if key in CAPTURE_PATH_KEYS:
            raise ValueError(f"capture output paths require --diagnostic-capture: {key}")
        _validate_environment_value(key, value)
        if key in overrides:
            raise ValueError(f"duplicate native environment key: {key}")
        overrides[key] = value
    return overrides


def make_contract(overrides: Mapping[str, str] | None = None) -> dict[str, object]:
    overrides = dict(overrides or {})
    parsed = parse_overrides([f"{key}={value}" for key, value in overrides.items()])
    effective = {
        key: ({"set": True, "value": parsed[key]} if key in parsed
              else {"set": False, "value": None})
        for key in NATIVE_ENV_KEYS
    }
    contract: dict[str, object] = {"schema": CONTRACT_SCHEMA,
                                  "keys": list(NATIVE_ENV_KEYS),
                                  "effective": effective}
    contract["sha256"] = _contract_digest(contract)
    return contract


def _diagnostic_contract_payload(contract: Mapping[str, object]) -> dict[str, object]:
    return {"schema": contract.get("schema"), "keys": contract.get("keys"),
            "effective": contract.get("effective"),
            "case_directory": contract.get("case_directory"),
            "capture_prefixes": contract.get("capture_prefixes")}


def _diagnostic_contract_digest(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(_diagnostic_contract_payload(contract))).hexdigest()


def make_diagnostic_contract(overrides: Mapping[str, str] | None, case_directory: os.PathLike[str] | str,
                             capture_prefixes: Mapping[str, os.PathLike[str] | str]) -> dict[str, object]:
    """Make a diagnostic-only contract from generated case-local prefixes."""
    parsed = parse_overrides([f"{key}={value}" for key, value in dict(overrides or {}).items()])
    case = Path(case_directory)
    if not case.is_absolute() or case.is_symlink() or not case.is_dir():
        raise ValueError("diagnostic case directory must be a fresh, existing absolute directory")
    root = case.resolve(strict=True)
    prefixes: dict[str, str] = {}
    for key, raw_prefix in sorted(capture_prefixes.items()):
        if key not in CAPTURE_PATH_KEYS:
            raise ValueError(f"unknown diagnostic capture path key: {key}")
        prefix = Path(raw_prefix)
        if not prefix.is_absolute() or prefix.is_symlink():
            raise ValueError(f"diagnostic prefix for {key} must be an absolute nonsymlink path")
        expected_relative = Path("diagnostics") / DIAGNOSTIC_CAPTURE_PREFIX_NAMES[key] / "capture"
        if prefix != root / expected_relative:
            raise ValueError(f"diagnostic prefix for {key} is not the fixed case-local prefix")
        parent = prefix.parent
        if parent.is_symlink() or not parent.is_dir() or prefix.exists():
            raise ValueError(f"diagnostic prefix for {key} is not fresh")
        prefixes[key] = str(prefix)
    effective: dict[str, dict[str, object]] = {}
    for key in NATIVE_ENV_KEYS:
        if key in prefixes:
            effective[key] = {"set": True, "value": prefixes[key]}
        elif key in parsed:
            effective[key] = {"set": True, "value": parsed[key]}
        else:
            effective[key] = {"set": False, "value": None}
    contract: dict[str, object] = {
        "schema": DIAGNOSTIC_CONTRACT_SCHEMA,
        "keys": list(NATIVE_ENV_KEYS),
        "effective": effective,
        "case_directory": str(root),
        "capture_prefixes": prefixes,
    }
    contract["sha256"] = _diagnostic_contract_digest(contract)
    return contract


def validate_contract(contract: object) -> dict[str, object]:
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("missing or unsupported effective native environment contract")
    if set(contract) != {"schema", "keys", "effective", "sha256"}:
        raise ValueError("effective native environment contract has ambiguous fields")
    if contract.get("keys") != list(NATIVE_ENV_KEYS):
        raise ValueError("effective native environment key registry differs from this source")
    effective = contract.get("effective")
    if not isinstance(effective, dict) or set(effective) != set(NATIVE_ENV_KEYS):
        raise ValueError("effective native environment must state every registered key")
    for key in NATIVE_ENV_KEYS:
        state = effective[key]
        if (not isinstance(state, dict) or type(state.get("set")) is not bool
                or set(state) != {"set", "value"}):
            raise ValueError(f"invalid absent/set representation for {key}")
        if state["set"]:
            if not isinstance(state["value"], str) or "\x00" in state["value"]:
                raise ValueError(f"set environment value for {key} must be a NUL-free string")
            if key in CAPTURE_PATH_KEYS:
                raise ValueError(f"capture/output path cannot be qualified: {key}")
            _validate_environment_value(key, state["value"])
        elif state["value"] is not None:
            raise ValueError(f"absent environment value for {key} must be null")
    expected = _contract_digest(contract)
    if contract.get("sha256") != expected:
        raise ValueError("effective native environment contract digest mismatch")
    return contract


def validate_diagnostic_contract(contract: object) -> dict[str, object]:
    if not isinstance(contract, dict) or contract.get("schema") != DIAGNOSTIC_CONTRACT_SCHEMA:
        raise ValueError("missing or unsupported diagnostic environment contract")
    if set(contract) != {"schema", "keys", "effective", "case_directory", "capture_prefixes", "sha256"}:
        raise ValueError("diagnostic environment contract has ambiguous fields")
    if contract.get("keys") != list(NATIVE_ENV_KEYS):
        raise ValueError("diagnostic environment key registry differs from this source")
    effective = contract.get("effective")
    if not isinstance(effective, dict) or set(effective) != set(NATIVE_ENV_KEYS):
        raise ValueError("diagnostic environment must state every registered key")
    case_text = contract.get("case_directory")
    if not isinstance(case_text, str) or not Path(case_text).is_absolute():
        raise ValueError("diagnostic contract case directory must be absolute")
    case = Path(case_text)
    if case.is_symlink() or not case.is_dir() or str(case.resolve(strict=True)) != case_text:
        raise ValueError("diagnostic contract case directory is missing, aliased, or symlinked")
    diagnostics_root = case / "diagnostics"
    if (diagnostics_root.is_symlink() or not diagnostics_root.is_dir() or
            diagnostics_root.resolve(strict=True) != diagnostics_root):
        raise ValueError("diagnostic contract output root is missing, aliased, or symlinked")
    prefix_map = contract.get("capture_prefixes")
    if not isinstance(prefix_map, dict) or not set(prefix_map).issubset(CAPTURE_PATH_KEYS):
        raise ValueError("diagnostic capture-prefix registry is invalid")
    expected_keys = set(prefix_map)
    for key in NATIVE_ENV_KEYS:
        state = effective[key]
        if (not isinstance(state, dict) or type(state.get("set")) is not bool
                or set(state) != {"set", "value"}):
            raise ValueError(f"invalid diagnostic absent/set representation for {key}")
        if not state["set"]:
            if state["value"] is not None or key in expected_keys:
                raise ValueError(f"invalid absent diagnostic value for {key}")
            continue
        value = state["value"]
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"set diagnostic value for {key} must be a NUL-free string")
        if key in CAPTURE_PATH_KEYS:
            if key not in expected_keys:
                raise ValueError(f"unregistered diagnostic capture path: {key}")
            relative = Path("diagnostics") / DIAGNOSTIC_CAPTURE_PREFIX_NAMES[key] / "capture"
            expected = case / relative
            if prefix_map[key] != str(expected) or value != str(expected):
                raise ValueError(f"diagnostic capture path for {key} is not the fixed case-local prefix")
            if (expected.parent.is_symlink() or not expected.parent.is_dir() or
                    expected.parent.resolve(strict=True) != expected.parent):
                raise ValueError(f"diagnostic capture prefix for {key} is not fresh")
        else:
            _validate_environment_value(key, value)
    if _diagnostic_contract_digest(contract) != contract.get("sha256"):
        raise ValueError("diagnostic environment contract digest mismatch")
    return contract


def validate_any_contract(contract: object) -> dict[str, object]:
    if isinstance(contract, dict) and contract.get("schema") == DIAGNOSTIC_CONTRACT_SCHEMA:
        return validate_diagnostic_contract(contract)
    return validate_contract(contract)


def is_qualification_run_record(record: object) -> bool:
    return (isinstance(record, dict) and record.get("schema") == RUN_SCHEMA and
            record.get("producer") in QUALIFIED_PRODUCERS and
            isinstance(record.get("contract"), dict) and
            record["contract"].get("schema") == CONTRACT_SCHEMA)


def child_environment(parent: Mapping[str, str], contract: object) -> tuple[dict[str, str], list[str]]:
    contract = validate_any_contract(contract)
    removed = sorted(key for key in parent if key.startswith(("NM_FEM_", "NM_CONTACT_")) or key in TIMING_KEYS)
    child = {key: value for key, value in parent.items()
             if not key.startswith(("NM_FEM_", "NM_CONTACT_")) and key not in TIMING_KEYS}
    for key, state in contract["effective"].items():
        if state["set"]:
            child[key] = state["value"]
    return child, removed


def create_diagnostic_capture_prefixes(case_directory: os.PathLike[str] | str,
                                       capture_keys: Sequence[str]) -> dict[str, str]:
    """Create a fresh, exclusive case-local directory for each selected capture."""
    case = Path(case_directory)
    if not case.is_absolute() or case.is_symlink() or not case.is_dir():
        raise ValueError("diagnostic case directory must be a fresh absolute directory")
    case = case.resolve(strict=True)
    keys = list(capture_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate diagnostic capture path key")
    if any(key not in CAPTURE_PATH_KEYS for key in keys):
        raise ValueError("diagnostic capture key is not a registered native output path")
    diagnostics = case / "diagnostics"
    diagnostics.mkdir(mode=0o700, exist_ok=False)
    if diagnostics.is_symlink() or not diagnostics.is_dir():
        raise ValueError("diagnostic output root is not a fresh directory")
    prefixes: dict[str, str] = {}
    for key in sorted(keys):
        directory = diagnostics / DIAGNOSTIC_CAPTURE_PREFIX_NAMES[key]
        directory.mkdir(mode=0o700, exist_ok=False)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"diagnostic output directory for {key} is not fresh")
        prefix = directory / "capture"
        if prefix.exists() or prefix.is_symlink():
            raise ValueError(f"diagnostic output prefix for {key} already exists")
        prefixes[key] = str(prefix)
    return prefixes



def _pending_checkpoint_capture(case: Path, key: str,
                                native_argv: Sequence[str], native_exit_code: int | None,
                                native_timed_out: bool,
                                effective_contract: Mapping[str, object] | None) -> dict[str, object]:
    """Prove a requested capture is strictly after a saved accepted prefix.

    Only the two explicit selectors used by coupon continuation are supported.
    A normal completed/failed run or an already-visited target cannot use this
    exception. The returned status is diagnostic-only, never a missing-capture
    success or a completed loading history.
    """
    if type(native_timed_out) is not bool:
        raise ValueError("native timeout state is missing or invalid")
    if native_timed_out:
        raise ValueError("timed-out native process cannot prove an accepted checkpoint boundary")
    if native_exit_code != 0 or effective_contract is None:
        raise ValueError("missing diagnostic capture has no successful accepted checkpoint boundary")
    validate_diagnostic_contract(effective_contract)
    if effective_contract.get("schema") != DIAGNOSTIC_CONTRACT_SCHEMA:
        raise ValueError("pending checkpoint capture requires a diagnostic contract")
    if (effective_contract.get("case_directory") != str(case)
            or key not in effective_contract["capture_prefixes"]):
        raise ValueError("pending checkpoint capture contract belongs to another case/path")
    flag = "--coupon-checkpoint-after-control"
    indices = [i for i, value in enumerate(native_argv) if value == flag]
    if len(indices) != 1 or indices[0]+1 >= len(native_argv):
        raise ValueError("missing diagnostic capture has no explicit coupon checkpoint request")
    raw_control = native_argv[indices[0]+1]
    if not re.fullmatch(r"[0-9]+", raw_control) or int(raw_control) >= UINT32_MAX:
        raise ValueError("invalid coupon checkpoint control")
    control = int(raw_control)
    output = case / "output"
    if output.is_symlink() or not output.is_dir() or output.resolve(strict=True) != output:
        raise ValueError("coupon checkpoint output directory is aliased")
    def regular_file(relative: str, cap: int) -> Path:
        if not isinstance(relative, str):
            raise ValueError("coupon checkpoint reference must be a relative path string")
        rel = Path(relative)
        path = output / rel
        if (rel.is_absolute() or not rel.parts or ".." in rel.parts or path.is_symlink()
                or not path.is_file() or path.resolve(strict=True) != path):
            raise ValueError("coupon checkpoint reference escaped its output directory")
        stat = path.stat()
        if stat.st_nlink != 1 or not 0 < stat.st_size <= cap:
            raise ValueError("coupon checkpoint reference is not a bounded regular file")
        return path
    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                result.update(block)
        return result.hexdigest()
    result_path = regular_file("result.json", 64*1024)
    prefix_path = regular_file("coupon-checkpoint-prefix.json", 64*1024)
    result, prefix = (json.loads(p.read_text()) for p in (result_path, prefix_path))
    if (result.get("schema") != "numi.cardboard.probe-result.v1"
            or result.get("status") != "accepted_checkpoint_prefix_saved"
            or type(result.get("accepted_steps")) is not int
            or result["accepted_steps"] != control+1 or result.get("physical_validation") is not False
            or prefix.get("schema") != "numi.cardboard.coupon-checkpoint-prefix.v1"
            or prefix.get("status") != "accepted_checkpoint_prefix_saved"
            or type(prefix.get("accepted_control_index")) is not int
            or prefix["accepted_control_index"] != control
            or type(prefix.get("next_control_index")) is not int
            or prefix["next_control_index"] != control+1
            or any(prefix.get(k) is not False for k in
                   ("full_protocol_complete", "trajectory_parity_established", "physical_validation"))):
        raise ValueError("missing capture is not bounded by an incomplete accepted checkpoint prefix")
    checkpoint = regular_file(prefix.get("checkpoint_file", ""), 2*1024**3)
    metadata = regular_file(prefix.get("metadata_file", ""), 64*1024)
    if (digest(checkpoint) != prefix.get("checkpoint_sha256")
            or digest(metadata) != prefix.get("metadata_sha256")):
        raise ValueError("accepted checkpoint prefix reference hash mismatch")
    values = {}
    for line in metadata.read_text().splitlines():
        name, separator, value = line.partition("=")
        if not separator or name in values:
            raise ValueError("accepted checkpoint metadata has malformed/duplicate fields")
        values[name] = value
    if (values.get("schema") not in ("numi.cardboard.accepted-coupon-resume.v1",
                                     "numi.cardboard.accepted-coupon-resume.v2")
            or values.get("accepted_control_index") != str(control)
            or values.get("next_control_index") != str(control+1)
            or values.get("both_environments_accepted") != "true"
            or values.get("checkpoint_sha256") != prefix["checkpoint_sha256"]
            or values.get("checkpoint_bytes") != str(checkpoint.stat().st_size)):
        raise ValueError("accepted checkpoint metadata does not certify this prefix boundary")
    selectors = {
        "NM_FEM_OPERATOR_FD_CAPTURE_PATH": ("NM_FEM_OPERATOR_FD_SELECTOR", 2),
        "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH": ("NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", 6),
    }
    if key not in selectors:
        raise ValueError("this missing capture selector is not supported by coupon continuation")
    selector_key, fields = selectors[key]
    state = effective_contract["effective"][selector_key]
    raw = state.get("value") if state.get("set") else None
    parts = raw.split(":") if isinstance(raw, str) else []
    if (len(parts) != fields or any(not re.fullmatch(r"[0-9]+", p) or int(p)>UINT32_MAX for p in parts)
            or int(parts[0]) <= control):
        raise ValueError("missing diagnostic capture target is not strictly after the saved prefix")
    return {"key": key, "selector_key": selector_key, "selector": raw,
            "target_control": int(parts[0]), "accepted_control": control,
            "status": "not_yet_visited_at_accepted_checkpoint",
            "result_sha256": digest(result_path), "prefix_report_sha256": digest(prefix_path),
            "checkpoint_sha256": prefix["checkpoint_sha256"],
            "metadata_sha256": prefix["metadata_sha256"], "positive_qualification_allowed": False}

def validate_diagnostic_outputs(case_directory: os.PathLike[str] | str,
                                capture_prefixes: Mapping[str, str], *,
                                native_argv: Sequence[str] = (), native_exit_code: int | None = None,
                                native_timed_out: bool = False,
                                effective_contract: Mapping[str, object] | None = None) -> dict[str, object]:
    """Validate only new regular files under each fixed prefix and enforce byte caps."""
    if type(native_timed_out) is not bool:
        raise ValueError("native timeout state is missing or invalid")
    case = Path(case_directory)
    if not case.is_absolute() or case.is_symlink() or not case.is_dir():
        raise ValueError("diagnostic case directory is invalid")
    root = case.resolve(strict=True)
    diagnostics_root = root / "diagnostics"
    if (diagnostics_root.is_symlink() or not diagnostics_root.is_dir() or
            diagnostics_root.resolve(strict=True) != diagnostics_root):
        raise ValueError("diagnostic output root is missing, aliased, or symlinked")
    outputs: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    total = 0
    if not capture_prefixes:
        return {"status": "no_capture_paths_requested", "files": [], "total_bytes": 0}
    for key, raw_prefix in sorted(capture_prefixes.items()):
        if key not in CAPTURE_PATH_KEYS:
            raise ValueError(f"unknown diagnostic capture key: {key}")
        expected = root / "diagnostics" / DIAGNOSTIC_CAPTURE_PREFIX_NAMES[key] / "capture"
        if (raw_prefix != str(expected) or expected.parent.is_symlink() or
                not expected.parent.is_dir() or expected.parent.resolve(strict=True) != expected.parent):
            raise ValueError(f"diagnostic prefix for {key} escaped its fixed case-local directory")
        entries = sorted(expected.parent.iterdir(), key=lambda item: item.name)
        if not entries:
            if not native_argv:
                raise ValueError(f"diagnostic capture produced no files for {key}")
            pending.append(_pending_checkpoint_capture(
                root, key, native_argv, native_exit_code, native_timed_out, effective_contract))
            continue
        if len(entries) > MAX_DIAGNOSTIC_CAPTURE_FILES:
            raise ValueError(f"diagnostic capture file-count cap exceeded for {key}")
        for path in entries:
            if not (path.name == expected.name or path.name.startswith(expected.name + ".") or
                    path.name.startswith(expected.name + "-")):
                raise ValueError(f"diagnostic output filename is outside its prefix for {key}")
            st = path.lstat()
            if path.is_symlink() or not path.is_file() or st.st_nlink != 1:
                raise ValueError(f"diagnostic output is not a new regular unlinked file for {key}")
            if st.st_size < 0 or st.st_size > MAX_DIAGNOSTIC_CAPTURE_FILE_BYTES:
                raise ValueError(f"diagnostic output file byte cap exceeded for {key}")
            total += st.st_size
            if total > MAX_DIAGNOSTIC_CAPTURE_TOTAL_BYTES:
                raise ValueError("diagnostic aggregate output byte cap exceeded")
            digest = hashlib.sha256()
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino, current.st_size) != (st.st_dev, st.st_ino, st.st_size):
                    raise ValueError(f"diagnostic output changed during validation for {key}")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            finally:
                os.close(fd)
            outputs.append({"key": key, "path": str(path.relative_to(root)),
                            "bytes": st.st_size, "sha256": digest.hexdigest()})
    return {"status": "validated_with_pending_checkpoint_targets" if pending else "validated",
            "pending_captures": pending, "files": outputs, "total_bytes": total,
            "file_count": len(outputs),
            "aggregate_cap_bytes": MAX_DIAGNOSTIC_CAPTURE_TOTAL_BYTES,
            "per_file_cap_bytes": MAX_DIAGNOSTIC_CAPTURE_FILE_BYTES}


def make_run_record(contract: object, *, producer: str, removed_inherited_keys: Sequence[str],
                    prediction_sha256: str, plan_sha256: str | None = None,
                    case_id: str | None = None) -> dict[str, object]:
    contract = validate_contract(contract)
    if not re.fullmatch(r"[0-9a-f]{64}", prediction_sha256):
        raise ValueError("prediction_sha256 must be a lowercase SHA-256 digest")
    removed = sorted(set(removed_inherited_keys))
    if any(not (key.startswith(("NM_FEM_", "NM_CONTACT_")) or key in TIMING_KEYS) for key in removed):
        raise ValueError("launch record may list only scrubbed native control key names")
    return {"schema": RUN_SCHEMA, "contract": contract,
            "contract_sha256": contract["sha256"],
            "prediction_sha256": prediction_sha256,
            "producer": producer,
            "plan_sha256": plan_sha256, "case_id": case_id,
            "launch": {"native_controls_scrubbed": True,
                       "removed_inherited_key_names": removed}}


def make_diagnostic_run_record(contract: object, *, removed_inherited_keys: Sequence[str],
                               prediction_sha256: str) -> dict[str, object]:
    contract = validate_diagnostic_contract(contract)
    if not re.fullmatch(r"[0-9a-f]{64}", prediction_sha256):
        raise ValueError("prediction_sha256 must be a lowercase SHA-256 digest")
    removed = sorted(set(removed_inherited_keys))
    if any(not (key.startswith(("NM_FEM_", "NM_CONTACT_")) or key in TIMING_KEYS) for key in removed):
        raise ValueError("launch record may list only scrubbed native control key names")
    return {"schema": DIAGNOSTIC_RUN_SCHEMA, "contract": contract,
            "contract_sha256": contract["sha256"], "prediction_sha256": prediction_sha256,
            "producer": DIAGNOSTIC_PRODUCER, "mode": "diagnostic",
            "launch": {"native_controls_scrubbed": True,
                       "removed_inherited_key_names": removed}}


def validate_run_record(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError("missing or unsupported effective native environment run record")
    diagnostic = record.get("schema") == DIAGNOSTIC_RUN_SCHEMA
    if not diagnostic and record.get("schema") != RUN_SCHEMA:
        raise ValueError("missing or unsupported effective native environment run record")
    contract = (validate_diagnostic_contract(record.get("contract")) if diagnostic
                else validate_contract(record.get("contract")))
    if record.get("contract_sha256") != contract["sha256"]:
        raise ValueError("run record contract hash mismatch")
    if diagnostic:
        if record.get("producer") != DIAGNOSTIC_PRODUCER or record.get("mode") != "diagnostic":
            raise ValueError("diagnostic environment record has the wrong producer or mode")
    elif record.get("producer") not in QUALIFIED_PRODUCERS:
        raise ValueError("environment record producer is not an admitted source-bound launcher")
    if not isinstance(record.get("prediction_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["prediction_sha256"]):
        raise ValueError("run record lacks a valid prediction digest")
    launch = record.get("launch")
    if not isinstance(launch, dict) or launch.get("native_controls_scrubbed") is not True:
        raise ValueError("run record does not prove inherited native controls were scrubbed")
    removed = launch.get("removed_inherited_key_names")
    if (not isinstance(removed, list) or removed != sorted(set(removed))
            or any(not isinstance(key, str) or not (key.startswith(("NM_FEM_", "NM_CONTACT_")) or key in TIMING_KEYS)
                   for key in removed)):
        raise ValueError("invalid scrubbed-key-name record")
    return record


def validate_prediction_binding(record: Mapping[str, object], prediction_raw: bytes) -> None:
    try:
        prediction = json.loads(prediction_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"environment prediction is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(prediction, dict):
        raise ValueError("environment prediction root must be an object")
    diagnostic = record.get("schema") == DIAGNOSTIC_RUN_SCHEMA
    contract = (validate_diagnostic_contract(record.get("contract")) if diagnostic
                else validate_contract(record.get("contract")))
    if (prediction.get("effective_native_environment_contract") != contract
            or prediction.get("effective_native_environment_contract_sha256") != contract["sha256"]):
        raise ValueError("prediction does not contain the exact effective environment contract")
    producer = record.get("producer")
    if producer == DIAGNOSTIC_PRODUCER:
        if prediction.get("schema") != "numi.cardboard.native-diagnostic-prediction.v1":
            raise ValueError("diagnostic run record points to the wrong prediction schema")
        if prediction.get("run_mode") != "diagnostic":
            raise ValueError("diagnostic prediction has the wrong run mode")
        expected_delta = {key: state["value"] for key, state in contract["effective"].items()
                          if state["set"]}
        if prediction.get("environment") != expected_delta:
            raise ValueError("diagnostic prediction override list differs from effective contract")
        return
    if producer == "source-bound-native-runner":
        if prediction.get("schema") != "numi.cardboard.native-prediction.v1":
            raise ValueError("source-bound run record points to the wrong prediction schema")
        effective = contract["effective"]
        expected_delta = {key: state["value"] for key, state in effective.items() if state["set"]}
        if prediction.get("environment") != expected_delta:
            raise ValueError("source-bound prediction override list differs from effective contract")
    elif producer == "resolution-study":
        if prediction.get("schema") != "numi.cardboard.resolution-case-prediction.v1":
            raise ValueError("resolution-study run record points to the wrong prediction schema")
        if prediction.get("plan_sha256") != record.get("plan_sha256"):
            raise ValueError("resolution prediction plan digest differs from environment record")
        case = prediction.get("case")
        if not isinstance(case, dict) or case.get("id") != record.get("case_id"):
            raise ValueError("resolution prediction case identity differs from environment record")


def write_run_record_new(path: os.PathLike[str] | str, record: object) -> None:
    """Write an environment sidecar once; existing evidence is never reused."""
    record = validate_run_record(record)
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_run_record(paths: Sequence[os.PathLike[str] | str]) -> tuple[dict[str, object] | None, str | None]:
    existing = [os.fspath(path) for path in paths if os.path.isfile(path)]
    if not existing:
        return None, None
    if len(existing) != 1:
        raise ValueError("ambiguous native environment sidecar locations")
    try:
        with open(existing[0], encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read native environment sidecar: {exc}") from exc
    return validate_run_record(value), existing[0]
