"""Opt-in wrapper adding three construction-only zero-displacement LP bounds.

All matched-input validation and fitting remain delegated to frozen 490 -> 485
-> 458 -> 408. The only solver change is to the 408 source-safe trust bounds
returned for the exact frozen 461 candidate and three audited area blockers.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY_490 = D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
ENTRY_490_SHA256 = "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"
HELPER_496 = Path(__file__).with_name("area_blocker_zero_bounds_496.py")
FLAG = "--zero-bound-486-area-blockers-496"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pinned(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_485_loader(entry_490, validation, hook_states, helper_module):
    """Wrap the exact 458 install callback so 408 bounds are patched before fit."""
    original_load_module = entry_490._load_module
    runner_485_path = Path(entry_490.RUNNER_485).resolve()
    def load_module(path):
        module = original_load_module(path)
        if Path(path).resolve() != runner_485_path:
            return module
        original_load = module.load
        def load(name, load_path):
            loaded = original_load(name, load_path)
            if name != "matched458_for_485":
                return loaded
            original_install = loaded.install_objective
            def install(owner, objective, crossing):
                state = helper_module.install_owner_bounds(owner, validation=validation)
                hook_states.append(state)
                return original_install(owner, objective, crossing)
            loaded.install_objective = install
            return loaded
        module.load = load
        return module
    entry_490._load_module = load_module


def main():
    argv = list(sys.argv[1:])
    if argv.count(FLAG) != 1:
        raise ValueError(f"explicit opt-in {FLAG} is required exactly once")
    argv.remove(FLAG)
    if not ENTRY_490.is_file() or sha(ENTRY_490) != ENTRY_490_SHA256:
        raise ValueError("frozen 490 delegate path/hash mismatch")
    if not HELPER_496.is_file():
        raise ValueError("496 blocker helper is missing")
    helper_sha_before = sha(HELPER_496)
    helper = load_pinned("area_blocker_zero_bounds_496", HELPER_496)
    entry = load_pinned("matched_component_fit_490_for_496", ENTRY_490)
    remaining, config_path, output_dir, actual_inputs = entry.extract_fit_inputs(argv)
    # Check the actual existing 475-schema matched config before installing any hook.
    matched = entry.validate_matched_inputs(config_path, actual_inputs)
    if matched["inputs"]["candidate"]["sha256"] != helper.EXPECTED_CANDIDATE_SHA256:
        raise ValueError("496 blocker bounds are only valid for the pinned 486 base candidate")
    validation = helper.validate_blocker_candidate(
        actual_inputs["candidate"], actual_inputs["audit159"]
    )
    delegate_source_before = entry.validate_source_lock()
    hook_states = []
    patch_485_loader(entry, validation, hook_states, helper)
    sys.argv[:] = [sys.argv[0], *remaining]
    entry.main()
    if sha(ENTRY_490) != ENTRY_490_SHA256 or sha(HELPER_496) != helper_sha_before:
        raise RuntimeError("496 frozen wrapper/helper source changed during delegated fit")
    delegate_source_after = entry.validate_source_lock()
    if delegate_source_before != delegate_source_after:
        raise RuntimeError("490/485/458/408 delegated source lock changed during fit")
    if len(hook_states) != 1 or hook_states[0]["calls"] != 1 or len(hook_states[0]["receipts"]) != 1:
        raise RuntimeError("408 trust-bound hook did not run exactly once")
    report_path = Path(output_dir) / "report.json"
    report = json.loads(report_path.read_text())
    source_box = report.get("source_safe_box")
    bound_receipt = source_box.get("area_blocker_zero_bounds_496") if isinstance(source_box, dict) else None
    if not isinstance(bound_receipt, dict):
        raise RuntimeError("delegated 408 report is missing the blocker-bound receipt")
    report["area_blocker_zero_bounds_496"] = {
        "classification": "explicit construction-only LP variable restriction; not runtime anchors/forces",
        "opt_in_flag": FLAG,
        "validated_blocker_inputs": validation,
        "runtime_hook_receipt": bound_receipt,
        "frozen_490_path": str(ENTRY_490), "frozen_490_sha256": ENTRY_490_SHA256,
        "helper_path": str(HELPER_496), "helper_sha256": helper_sha_before,
        "native_admission": False, "accepted_physical_steps": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
