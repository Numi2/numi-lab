"""Isolated v4 runner: bind the parity evaluator to the loaded 541 module; v2/v3 remain unchanged."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
TOPOLOGY = ROOT / "D/surface-clearance-546/topology/topology-bone-aware-025.py"
CHAIN = ROOT / "D/surface-clearance-546/topology/chain_exact_flip_batches_025.py"
OWNER541 = ROOT / "D/focused-clearance-541/topology/repair_focused_clearance_541.py"
OWNER370 = ROOT / "D/connected-remesh-reuse-370/select_bone_reducing_flips_370.py"
POLICY = HERE / "quality_policy025_variant.py"
EXPECTED_TOPOLOGY_SHA = "9123965621161beb6408f6c9663d8d36968cf1113c3a97106e968ea1c754d84e"
EXPECTED_541_SHA = "26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa"
EXPECTED_370_SHA = "c8650b8c311ca8f386ebf1713b5fc285b7bded860cfa878192472704ac1336d5"
EXPECTED_CHAIN_SHA = "d42db71f6003f60eec69787156775e2133be2c54f69e37047ea8c451c76bd5a9"
EXPECTED_POLICY_SHA = "02078d7a942770aae7544beb479d7dd633ad2a7f514e1239ce73ce51c253784e"
POLICY_NAME = "quality-025-relative-angle-loss-and-exact-reduction-v2-parity-bound-v4"
CHAIN_MODE = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy_record():
    return {
        "name": POLICY_NAME,
        "minimum_fixed_reference_area_ratio": 0.1,
        "minimum_fixed_reference_normal_cosine_exclusive": 0.0,
        "minimum_patch_angle_deg": 2.0,
        "maximum_min_angle_loss_deg": "min(5.0, 0.5 * prior two-face minimum angle)",
        "quality_gain_gate_before_exact_pair_scoring": False,
        "exact_185_bank_no_increase_and_strict_total_reduction_required": True,
        "batch_exact_185_replacement_strict_reduction_required": True,
        "self_safety_408_315_incremental_check_unchanged": True,
        "402_405_topology_ancestry_and_coordinate_gates_unchanged": True,
        "full_130_159_audits_still_required": True,
    }


def source_pins():
    pins = {
        "topology_025": {"path": str(TOPOLOGY), "sha256": sha(TOPOLOGY)},
        "owner_541": {"path": str(OWNER541), "sha256": sha(OWNER541)},
        "owner_370": {"path": str(OWNER370), "sha256": sha(OWNER370)},
        "policy_helper": {"path": str(POLICY), "sha256": sha(POLICY)},
        "variant_runner_v4": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
    }
    if CHAIN_MODE:
        pins["chain_025"] = {"path": str(CHAIN), "sha256": sha(CHAIN)}
    return pins


def verify_frozen_sources(chain_mode=False):
    expected = {
        TOPOLOGY: EXPECTED_TOPOLOGY_SHA,
        OWNER541: EXPECTED_541_SHA,
        OWNER370: EXPECTED_370_SHA,
        POLICY: EXPECTED_POLICY_SHA,
    }
    if chain_mode:
        expected[CHAIN] = EXPECTED_CHAIN_SHA
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"pinned topology source changed: {path}")


def write_json_roundtrip(path, value):
    """Write canonical JSON with a real final newline and validate the round trip."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded)
    actual_bytes = path.read_bytes()
    actual = json.loads(actual_bytes)
    if actual != value:
        raise ValueError("JSON annotation did not round-trip")
    if not actual_bytes.endswith(b"\n") or actual_bytes.endswith(b"\\n"):
        raise ValueError("JSON output must end with one newline, not a literal backslash-n")
    return actual


def annotate_report(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    report["quality_policy_025_variant"] = {
        "policy": policy_record(),
        "source_pins": source_pins(),
        "fast_and_parity_evaluators_share_helper": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write_json_roundtrip(report_path, report)
    return report


def bind_parity_owner(owner_module, policy_module):
    """Bind the owner module into the five-argument 541 evaluator interface."""
    def evaluate(c, op, quality_before, points_mm, ref_mm):
        return policy_module.evaluate_op_parity(
            owner_module, c, op, quality_before, points_mm, ref_mm
        )
    owner_module.evaluate_op = evaluate
    return evaluate


def run_topology():
    verify_frozen_sources(False)
    policy = load_module("quality_policy025_variant_runtime_v4", POLICY)
    topology = load_module("topology_025_policy_variant_runtime_v4", TOPOLOGY)
    installed = policy.install_policy_hooks(topology, policy, OWNER541)
    for owner_module in installed:
        bind_parity_owner(owner_module, policy)
    topology.main()
    if len(installed) != 1:
        raise RuntimeError(f"expected one patched 541 evaluator, got {len(installed)}")
    report_path = Path(sys.argv[sys.argv.index("--output-dir") + 1]) / "report.json"
    annotate_report(report_path)
    verify_frozen_sources(False)


def run_chain():
    global CHAIN_MODE
    verify_frozen_sources(True)
    if "--chain" not in sys.argv:
        raise AssertionError("internal chain dispatch error")
    sys.argv.remove("--chain")
    CHAIN_MODE = True
    chain = load_module("chain_025_policy_variant_runtime_v4", CHAIN)
    chain.WRAPPER = Path(__file__).resolve()
    chain.WRAPPER_SHA = sha(__file__)
    chain.main()
    output_root = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
    receipt = {
        "schema": "numi.human.topology_025_quality_policy_chain.v4",
        "status": "BOUNDED_CHAIN_COMPLETE_PENDING_ONE_FINAL_FULL_EXACT_AUDIT",
        "policy": policy_record(),
        "source_pins": source_pins(),
        "variant_wrapper_sha256": sha(__file__),
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    write_json_roundtrip(output_root / "quality-policy-variant-receipt.json", receipt)
    verify_frozen_sources(True)


def main():
    if "--chain" in sys.argv:
        run_chain()
    else:
        run_topology()


if __name__ == "__main__":
    main()
