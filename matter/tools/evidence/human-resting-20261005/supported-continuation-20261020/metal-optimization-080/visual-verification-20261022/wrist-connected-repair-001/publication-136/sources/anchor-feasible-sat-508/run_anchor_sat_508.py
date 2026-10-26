"""508 opt-in anchor-compatible SAT wrapper over frozen 505 -> 490 -> 485 -> 458."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY_505 = D / "D/D/area-blocker-zero-bounds-505/run_matched_fit_505.py"
HELPER_505 = D / "D/D/area-blocker-zero-bounds-505/area_blocker_zero_bounds_505.py"
ENTRY_490 = D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
HELPER_508 = Path(__file__).with_name("anchor_feasible_sat_508.py")
SAT_466 = D / "D/D/triangle-sat-rows-466/triangle_sat_rows_466.py"
FLAG = "--anchor-feasible-sat-508"
ALL_FLAG = "--anchor-feasible-sat-all-pairs-508"
EXPECTED = {
    ENTRY_505: "6509b68e4aabea0b8d11c499c7a6ef426137bb90459fedbc7e005300be14312b",
    HELPER_505: "993ad02efc1447449acdee72da6afee821333bd9badee792e9821efa22e74dae",
    ENTRY_490: "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693",
    SAT_466: "cf4df8cd8a3f2773f0992a5f9a1d9b899d64db781b0a58cc2ed618285fa11cf3",
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    argv = list(sys.argv[1:])
    if argv.count(FLAG) != 1:
        raise ValueError(f"explicit opt-in {FLAG} is required exactly once")
    argv.remove(FLAG)
    all_pairs = argv.count(ALL_FLAG) == 1
    if argv.count(ALL_FLAG) > 1:
        raise ValueError(f"{ALL_FLAG} may appear at most once")
    if all_pairs:
        argv.remove(ALL_FLAG)
    if not all(Path(p).is_file() for p in (ENTRY_505, HELPER_505, ENTRY_490, HELPER_508, SAT_466)):
        raise FileNotFoundError("508 pinned source is missing")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"508 frozen delegate source hash mismatch: {path}")
    pins = {str(p): sha(p) for p in (*EXPECTED, HELPER_508, Path(__file__))}
    entry505 = load("fit505_for_508", ENTRY_505)
    helper = load("anchor_feasible_sat_508", HELPER_508)
    sat = load("triangle_sat_rows_466_for_508", SAT_466)
    original_patch = entry505.patch_485_loader
    state = {"calls": 0, "selected_pair_count": 0, "all_pairs": all_pairs, "axes": []}

    def patch_485_loader(entry490, validation, hook_states, helper505):
        original_patch(entry490, validation, hook_states, helper505)
        load_module_505 = entry490._load_module
        runner485 = Path(entry490.RUNNER_485).resolve()
        def load_module(path):
            module = load_module_505(path)
            if Path(path).resolve() != runner485:
                return module
            old_load = module.load
            def load_458(name, load_path):
                loaded = old_load(name, load_path)
                if name != "matched458_for_485":
                    return loaded
                old_install = loaded.install_objective
                def install(owner, objective, crossing):
                    def build(*args, **kwargs):
                        baseline = objective.build_fixed_bone_outward_plane_rows(*args, **kwargs)
                        diag = baseline["diagnostics"]
                        if state["all_pairs"]:
                            selected = {(int(p["face_row"]), int(p["body"]), int(p["bone_id"]), int(p["triangle_id"]))
                                        for p in crossing["pairs"]}
                        else:
                            selected = {tuple(map(int, row["identity"]))
                                        for row in diag["anchored_positive_lowers"]}
                        if not selected:
                            state["calls"] += 1
                            state["selected_pair_count"] = 0
                            return baseline
                        points_mm, faces, support, anchors = args[:4]
                        points_mm, faces, support, anchors = args[:4]
                        result = helper.build_anchor_compatible_rows(
                            points_mm, faces, support, anchors, crossing["pairs"],
                            crossing["bone_triangles_mm"], sat=sat,
                            selected_pair_ids=selected,
                            total_weight=float(kwargs.get("total_weight", 1.0)))
                        if result["diagnostics"]["anchored_positive_lower_count"]:
                            raise ValueError("508 selected plane still requires positive anchored displacement")
                        state["calls"] += 1
                        state["selected_pair_count"] = len(selected)
                        state["axes"] = result["diagnostics"]["anchor_compatible_axes"]
                        return result
                    proxy = types.SimpleNamespace(build_fixed_bone_outward_plane_rows=build)
                    return old_install(owner, proxy, crossing)
                loaded.install_objective = install
                return loaded
            module.load = load_458
            return module
        entry490._load_module = load_module

    entry505.patch_485_loader = patch_485_loader
    sys.argv[:] = [sys.argv[0], *argv]
    entry505.main()
    report_path = Path(argv[argv.index("--output") + 1]) / "report.json"
    report = json.loads(report_path.read_text())
    if state["calls"] != 1:
        raise RuntimeError("508 objective hook did not run exactly once")
    report["anchor_compatible_sat_508"] = {
        "classification": "opt-in construction objective; all final exact audits remain mandatory",
        "opt_in_flag": FLAG,
        "all_pairs_flag": bool(all_pairs),
        "selection": "all crossed pairs" if all_pairs else "only pairs with positive anchored corner deficits under 458",
        "selected_pair_count": state["selected_pair_count"],
        "selected_axes": state["axes"],
        "all_selected_anchors_compatible": True,
        "same_point_target_prefix": True,
        "same_505_zero_bounds_and_original_hard_guards": True,
        "source_pins": pins,
        "global_clearance_certificate": False,
        "native_admission": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if any(sha(Path(path)) != digest for path, digest in pins.items()):
        raise RuntimeError("508 source changed during delegated fit")


if __name__ == "__main__":
    main()

