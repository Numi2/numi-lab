#!/usr/bin/env python3
"""One matched 128-child parity/timing sample for stacked clip and optional 085 separator."""
import ast, csv, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from plane_filter_085 import INSIDE_TOL, prepare_tetra
from plane_filter_085b import tetra_intersection_volume_stacked, filtered_tetra_intersection_volume_stacked

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAND = Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/whole-skin-fat-reference-084-candidate.npz")
NODES = ROOT / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight/compiled-skin-fem-nodes.csv"
TETS = ROOT / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight/compiled-skin-fem-tets.csv"
OWNER = ROOT / "connected-skin-parent-mass-transfer-067/native-parent-transfer-073/build_native_parent_transfer_073.py"
OUT = Path(__file__).resolve().parent / "sample-128b-report.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(newline="") as f:
        return list(csv.DictReader(f))


def exact_owner_clip():
    tree = ast.parse(OWNER.read_text())
    wanted = {"halfspaces", "tetra_intersection_volume"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {"np": np, "itertools": itertools, "ConvexHull": ConvexHull, "QhullError": QhullError}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(OWNER), "exec"), ns)
    return ns["tetra_intersection_volume"]


def tetvol(v):
    return abs(float(np.linalg.det(v[1:] - v[0]))) / 6.0


def main():
    begin = time.perf_counter()
    z = np.load(CAND, allow_pickle=False)
    nodes = read_csv(NODES)
    rows = [r for r in read_csv(TETS) if int(r["identity_active_bit"]) == 1]
    native_global = np.asarray([int(r["global_node"]) for r in nodes], dtype=np.int64)
    native_xyz = np.asarray([[float(r[k]) for k in ("rest_x", "rest_y", "rest_z")] for r in nodes])
    old_t = np.asarray([[int(r[f"source_node{j}"]) for j in range(4)] for r in rows], dtype=np.int64)
    old_xyz = native_xyz[old_t]
    new_t = z["tetrahedra_zero_based"].astype(np.int64)
    new_xyz = z["reference_points_m"][new_t]
    if len(nodes) != 7233 or len(rows) != 28632 or len(new_t) != 32801:
        raise RuntimeError("pinned source/candidate dimensions changed")
    if not np.array_equal(z["source_global_node_ids"].astype(np.int64), native_global):
        raise RuntimeError("084 original node identity prefix changed")
    if not np.array_equal(z["reference_points_m"][:len(nodes)], native_xyz):
        raise RuntimeError("084 reference prefix differs from native F32-promoted-F64 source")

    # Same deterministic evenly spaced 128-child subset as the earlier 085 sample.
    ids = np.unique(np.linspace(0, len(new_t) - 1, 128, dtype=np.int64))
    old_lo, old_hi = old_xyz.min(axis=1), old_xyz.max(axis=1)
    pair_rows = []
    for ci in ids:
        lo, hi = new_xyz[ci].min(axis=0), new_xyz[ci].max(axis=0)
        oi = np.flatnonzero(np.all(old_hi >= lo - INSIDE_TOL, axis=1) &
                            np.all(old_lo <= hi + INSIDE_TOL, axis=1))
        pair_rows.extend((int(ci), int(oj)) for oj in oi)

    # Prepared tolerance-expanded planes are shared across all paired tests.
    tprep = time.perf_counter()
    prep_old = [prepare_tetra(v) for v in old_xyz]
    prep_new = [prepare_tetra(v) for v in new_xyz]
    prep_seconds = time.perf_counter() - tprep

    clip = exact_owner_clip()
    t_base = time.perf_counter()
    base_values = np.asarray([clip(new_xyz[i], old_xyz[j]) for i, j in pair_rows])
    baseline_seconds = time.perf_counter() - t_base

    t_stack = time.perf_counter()
    stack_values = np.asarray([tetra_intersection_volume_stacked(new_xyz[i], old_xyz[j])
                               for i, j in pair_rows])
    stack_seconds = time.perf_counter() - t_stack

    t_filter = time.perf_counter()
    filtered = []
    rejected = []
    for i, j in pair_rows:
        value, was_rejected = filtered_tetra_intersection_volume_stacked(
            new_xyz[i], old_xyz[j], prepared_a=prep_new[i], prepared_b=prep_old[j])
        filtered.append(value)
        rejected.append(was_rejected)
    filtered = np.asarray(filtered); rejected = np.asarray(rejected, dtype=bool)
    filter_seconds = time.perf_counter() - t_filter

    child_vol = np.asarray([tetvol(new_xyz[i]) for i, _ in pair_rows])
    parent_vol = np.asarray([tetvol(old_xyz[j]) for _, j in pair_rows])
    overlap_threshold = np.maximum(np.minimum(child_vol, parent_vol) * 1e-13, 1e-25)
    base_pos = base_values > overlap_threshold
    stack_pos = stack_values > overlap_threshold
    filter_pos = filtered > overlap_threshold
    parity = {
        "stacked_clip_volumes_bitwise_equal_to_073": bool(np.array_equal(base_values, stack_values)),
        "stacked_positive_pair_mask_equal": bool(np.array_equal(base_pos, stack_pos)),
        "filtered_positive_pair_mask_equal": bool(np.array_equal(base_pos, filter_pos)),
        "filtered_unsafe_rejects": int(np.count_nonzero(rejected & base_pos)),
        "stacked_max_absolute_volume_delta_m3": float(np.max(np.abs(base_values-stack_values))) if len(pair_rows) else 0.0,
        "filtered_max_absolute_volume_delta_m3": float(np.max(np.abs(base_values-filtered))) if len(pair_rows) else 0.0,
    }
    counts = {
        "sample_children": int(len(ids)), "aabb_pairs": len(pair_rows),
        "plane_rejected": int(rejected.sum()), "exact_survivors": int((~rejected).sum()),
        "positive_overlap_owner_073": int(base_pos.sum()),
        "positive_overlap_stack_all_pairs": int(stack_pos.sum()),
        "positive_overlap_filtered": int(filter_pos.sum()),
    }
    report = {
        "status": "sample_complete",
        "owner_clip": "073 combination order, scalar determinant cutoff 1e-10, inside tolerance 2e-10 m, duplicate test and ConvexHull; 085b stacks the same systems; optional 085 plane rejection remains experimental",
        "input_sha256": {"candidate_084": sha(CAND), "native_nodes": sha(NODES),
                         "native_tets": sha(TETS), "owner_073": sha(OWNER)},
        "code_sha256": {"plane_filter_085b.py": sha(Path(__file__).with_name("plane_filter_085b.py")),
                        "plane_filter_085.py": sha(Path(__file__).with_name("plane_filter_085.py"))},
        "dimensions": {"candidate_tets": len(new_t), "native_active_tets": len(old_t),
                       "candidate_nodes": len(z["reference_points_m"])},
        "reference_frame": "084 F64 reference_points_m; original 7,233 rows bitwise equal to native compiled rest F32 promoted F64",
        "sample_ids": ids.tolist(), "counts": counts, "parity": parity,
        "seconds": {"prepare_all_planes": prep_seconds,
                    "unculled_073_scalar_clip": baseline_seconds,
                    "unculled_085b_stacked_clip": stack_seconds,
                    "filtered_085b_stacked_clip": filter_seconds,
                    "speedup_stacked_vs_073": baseline_seconds/stack_seconds if stack_seconds else None,
                    "speedup_filtered_vs_073": baseline_seconds/filter_seconds if filter_seconds else None,
                    "total": time.perf_counter()-begin},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

