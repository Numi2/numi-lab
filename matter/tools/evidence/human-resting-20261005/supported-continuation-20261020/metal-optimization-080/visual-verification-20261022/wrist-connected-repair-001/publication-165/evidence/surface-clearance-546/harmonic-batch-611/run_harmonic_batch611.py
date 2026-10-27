"""Fixed-seed bounded wrapper around frozen 577 harmonic patch search.
Construction proposal only; existing exact quality and final checks remain required.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D/surface-clearance-546"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
SLOT = Path("/tmp/numi-human-geometry-slot.py")
SEARCH577 = D / "harmonic-patch-577/search_harmonic577.py"
SEARCH577_SHA = "49ae65ae9a73004f535f1ec2735b4af77ab98af4c27a9444b37f92d7ba527f88"
SEED_DEFAULT = D / "harmonic-batch-611/seed-patches-602.json"
PAYLOAD_DEFAULT = D / "same-count-star-payload-568/same_count_patch_payload568.py"
LINEAGE566 = D / "affine-ancestry-refresh-566/rebuild_affine_rows566.py"
CHECK545 = ROOT / "D/contained-clearance-545/run_candidate_checks545.py"
FINALIZER546 = D / "finalize_existing_checks546.py"
START_MARK = "  neighbors=defaultdict(set)\n"
END_MARK = ";used_nodes=set();round_ops=0"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stage_module():
    path = D / "continue_surface_round546.py"
    spec = importlib.util.spec_from_file_location("stage611", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adapt_search_source(source: str) -> str:
    if source.count(START_MARK) != 1 or source.count(END_MARK) != 1:
        raise ValueError("frozen 577 center-generation block changed; refusing adapter")
    start = source.index(START_MARK)
    end = source.index(END_MARK, start) + len(END_MARK)
    if source.find(END_MARK) < start:
        raise ValueError("bad source block ordering")
    replacement = (
        "  centers=[tuple(map(int,rows)) for rows in FIXED_CENTER_ROWS]\n"
        "  used_nodes=set();round_ops=0"
    )
    return source[:start] + replacement + source[end:]


def fixed_rows_from_seed(seed: dict, faces: np.ndarray) -> list[list[int]]:
    if seed.get("schema") != "numi.human.harmonic_batch611_seed_patches.v1":
        raise ValueError("unexpected seed-list schema")
    if seed.get("status") != "INERT_VALIDATED_FACE_SEED_LIST_ONLY":
        raise ValueError("seed list is not the inert validated input")
    patches = seed.get("patches")
    if not isinstance(patches, list) or len(patches) != 17:
        raise ValueError("expected exactly 17 fixed patches")
    rows_out: list[list[int]] = []
    allrows: list[int] = []
    pair_incidences = 0
    for p in patches:
        rows = p.get("crossing_face_rows")
        if not isinstance(rows, list) or len(rows) != p.get("face_count") or not rows:
            raise ValueError("invalid crossing face row list")
        rr = [int(r) for r in rows]
        if len(set(rr)) != len(rr) or min(rr) < 0 or max(rr) >= len(faces):
            raise ValueError("duplicate/out-of-domain patch face rows")
        if p.get("face_kind") != 2 or p.get("validated_as_single_oriented_disk") is not True:
            raise ValueError("patch lacks validated skin-disk contract")
        if not (1 <= int(p.get("boundary_vertex_count", 0)) <= 48):
            raise ValueError("patch violates frozen boundary limit")
        patch_nodes = sorted(set(map(int, faces[np.asarray(rr, dtype=np.int64)].reshape(-1))))
        if patch_nodes != sorted(map(int, p.get("used_vertex_rows", []))):
            raise ValueError("patch vertex inventory mismatch")
        rows_out.append(rr)
        allrows.extend(rr)
        pair_incidences += int(p.get("target_pair_count", -1))
    if len(allrows) != 224 or len(set(allrows)) != 224 or pair_incidences != 703:
        raise ValueError("seed pair-incidence/unique-face counts changed")
    if seed.get("patch_count") != 17 or seed.get("covered_pair_incidences") != 703 or seed.get("unique_skin_faces") != 224:
        raise ValueError("seed count receipt mismatch")
    return rows_out


def validate_inputs(state_path: Path, seed_path: Path, payload: Path) -> tuple[dict, dict, dict, list[list[int]]]:
    state = read_json(state_path)
    seed = read_json(seed_path)
    if sha(state_path) != seed.get("baseline_state_sha256") or str(state_path) != seed.get("baseline_state"):
        raise ValueError("seed baseline state path/hash mismatch")
    for role in ("candidate", "surface_report", "point_summary"):
        p = Path(state[role])
        if sha(p) != state[role + "_sha256"]:
            raise ValueError("state role hash mismatch: " + role)
    if state["candidate"] != seed.get("candidate") or state["candidate_sha256"] != seed.get("candidate_sha256"):
        raise ValueError("seed candidate differs from state")
    if state["surface_report"] != seed.get("surface_report") or state["surface_report_sha256"] != seed.get("surface_report_sha256"):
        raise ValueError("seed surface report differs from state")
    if state["point_summary"] != seed.get("point_summary") or state["point_summary_sha256"] != seed.get("point_summary_sha256"):
        raise ValueError("seed point summary differs from state")
    expected = seed.get("source_pins", {})
    for p, digest in expected.items():
        if sha(p) != digest:
            raise ValueError("seed source pin mismatch: " + p)
    if sha(SEARCH577) != SEARCH577_SHA:
        raise ValueError("frozen 577 search source changed")
    if not payload.is_file():
        raise FileNotFoundError(payload)
    report = read_json(state["surface_report"])
    summary = read_json(state["point_summary"])
    if report.get("candidate_sha256") != state["candidate_sha256"] or report.get("complete") is not True or int(report.get("bank_count", -1)) != 185 or int(report.get("expected_bank_count", -1)) != 185:
        raise ValueError("baseline surface report is not complete/candidate-matched")
    if summary.get("candidate_sha256") != state["candidate_sha256"] or summary.get("complete_surface_audit") is not True:
        raise ValueError("baseline point summary is not complete/candidate-matched")
    if int(summary.get("exact_PLC_surface_pairs", -1)) != 0:
        raise ValueError("baseline is not exact-self clear")
    with np.load(state["candidate"], allow_pickle=False) as z:
        candidate = {k: np.array(z[k], copy=True) for k in z.files}
    faces = candidate["source_O_faces"]
    if faces.shape != (41733, 3) or candidate["points_world_m"].shape != (20889, 3):
        raise ValueError("candidate domain differs from validated seed scope")
    rows = fixed_rows_from_seed(seed, faces)
    return state, seed, candidate, rows


def run_search_worker(state_path: Path, seed_path: Path, payload: Path, output: Path, seconds: float) -> None:
    if sha(SEARCH577) != SEARCH577_SHA:
        raise ValueError("frozen 577 search source hash mismatch")
    _, _, _, rows = validate_inputs(state_path, seed_path, payload)
    original = SEARCH577.read_text()
    adapted = adapt_search_source(original)
    if str(SEARCH577.parent) not in sys.path:
        sys.path.insert(0, str(SEARCH577.parent))
    ns = {
        "__name__": "harmonic_search_611_in_memory",
        "__file__": str(SEARCH577),
        "FIXED_CENTER_ROWS": rows,
    }
    exec(compile(adapted, str(SEARCH577), "exec"), ns)
    args = SimpleNamespace(
        state=str(state_path), output=str(output), payload_helper=str(payload),
        seconds=float(seconds), rounds=1, max_centers=17, max_degree=48,
        max_candidates_per_center=128,
    )
    ns["run"](args)


def run_batch(state_path: Path, seed_path: Path, payload: Path, out: Path, seconds: float) -> None:
    if seconds <= 0 or seconds > 90:
        raise ValueError("611 proposal budget must be in (0, 90] seconds")
    if out.exists():
        raise FileExistsError(out)
    state, seed, _, _ = validate_inputs(state_path, seed_path, payload)
    out.mkdir(parents=True)
    wrapper_sha = sha(Path(__file__))
    source_list = [Path(__file__), SEARCH577, SEARCH577.parent / "harmonic_candidates577.py",
                   D / "regional-patch-571/regional_candidates571.py",
                   D / "connected-star-569/star_candidates569.py",
                   D / "changed-triangle-quality-572/changed_triangle_masks572.py",
                   D / "same-count-star-payload-568/cached_counter569_v2.py",
                   D / "same-count-star-payload-568/same_count_patch_payload568.py",
                   D / "topology/topology-bone-aware-025.py",
                   D / "topology-policy-root-v6/quality_policy025_variant.py",
                   D / "affine-ancestry-refresh-566/rebuild_affine_rows566.py",
                   D / "continue_surface_round546.py", seed_path, state_path,
                   Path(state["candidate"]), Path(state["surface_report"]), Path(state["point_summary"]), payload]
    pins = {str(p): sha(p) for p in source_list}
    launch = {
        "schema": "numi.human.harmonic_batch611_launch.v1",
        "wrapper": str(Path(__file__)), "wrapper_sha256": wrapper_sha,
        "state": str(state_path), "state_sha256": sha(state_path),
        "candidate_sha256": state["candidate_sha256"],
        "seed_list": str(seed_path), "seed_list_sha256": sha(seed_path),
        "fixed_patch_count": 17, "fixed_round_count": 1,
        "proposal_budget_seconds": float(seconds), "source_pins": pins,
        "construction_only": True, "native_admission": False, "accepted_physical_steps": 0,
    }
    write_json(out / "launch.json", launch)
    m = stage_module()
    m.run_stage(
        [PYTHON, Path(__file__), "--worker", "--state", state_path, "--seed", seed_path,
         "--payload-helper", payload, "--output", out / "search", "--seconds", str(seconds)],
        out / "search-stage", slot=True, source_paths=tuple(Path(p) for p in pins), timeout=600)
    if any(sha(p) != digest for p, digest in pins.items()):
        raise RuntimeError("611 source/input pin changed during search")
    result = read_json(out / "search/report.json")
    summary = {k: result.get(k) for k in ("status", "before_pairs", "after_pairs_composed", "reduction", "exact_triangle_calls", "seconds")}
    write_json(out / "search-summary.json", summary)
    if not result.get("accepted_operations"):
        write_json(out / "completion.json", {
            "status": "NO_ACCEPTED_TOPOLOGY_OPERATION", "search_report": str(out / "search/report.json"),
            "search_report_sha256": sha(out / "search/report.json"), "full_check_run": False,
            "native_admission": False, "accepted_physical_steps": 0,
        })
        return
    m.run_stage([PYTHON, LINEAGE566, "--candidate", out / "search/candidate.npz", "--out", out / "lineage"],
                out / "lineage-stage", slot=True, source_paths=(LINEAGE566,), timeout=90)
    m.run_stage([PYTHON, CHECK545, "--candidate", out / "lineage/candidate.npz", "--baseline-state", state_path,
                 "--trust-mm", "4", "--output-dir", out / "full-check"],
                out / "full-check-stage", slot=False, source_paths=(CHECK545,), timeout=900)
    check = read_json(out / "full-check/report.json")
    completion = {
        "status": check.get("status"), "search_report": str(out / "search/report.json"),
        "search_report_sha256": sha(out / "search/report.json"),
        "lineage_report": str(out / "lineage/report.json"), "lineage_report_sha256": sha(out / "lineage/report.json"),
        "full_check_report": str(out / "full-check/report.json"), "full_check_report_sha256": sha(out / "full-check/report.json"),
        "surface_pairs": check.get("surface_pairs"), "contained_pairs": check.get("contained_pairs"),
        "self_pairs": check.get("self_pairs"), "construction_progress": check.get("construction_progress"),
        "native_admission": False, "accepted_physical_steps": 0,
    }
    write_json(out / "completion.json", completion)
    if check.get("construction_progress") and check.get("contained_pairs") == 0 and check.get("self_pairs") == 0:
        m.run_stage([PYTHON, FINALIZER, "--candidate", out / "lineage/candidate.npz", "--baseline-state", state_path,
                     "--point-summary", out / "full-check/audit-130/summary.json",
                     "--surface-report", out / "full-check/audit-159/report.json",
                     "--out", out / "final", "--trust-mm", "4"],
                    out / "final-stage", slot=False, source_paths=(FINALIZER,), timeout=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--seed", type=Path, default=SEED_DEFAULT)
    ap.add_argument("--payload-helper", type=Path, default=PAYLOAD_DEFAULT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=90)
    args = ap.parse_args()
    if args.worker:
        run_search_worker(args.state, args.seed, args.payload_helper, args.output, args.seconds)
    else:
        run_batch(args.state, args.seed, args.payload_helper, args.output, args.seconds)

if __name__ == "__main__":
    main()
