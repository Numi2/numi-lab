#!/usr/bin/env python3
"""Read-only 637-parent proof for the 580 same-count topology proposal."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D")
ROOT = D / "surface-clearance-546"
SEARCH = ROOT / "connected-topology-580/run-010-from689/search"
BASE_STATE = ROOT / "component-sat-selector-638/guard-transfer-725-695/prep-002/axis-fit-695-001/component-retention-631-v2-695-001/final/state.json"
PARENT_DIR = ROOT / "box-aware-sat-fit-637/run-001-from766/merit-631-001"
PARENT = PARENT_DIR / "candidate.npz"
PARENT_STATE = PARENT_DIR / "final/state.json"
PAYLOAD = ROOT / "same-count-star-payload-568/same_count_patch_payload568.py"
OWNER557 = ROOT / "connected-reference-degeneracy-repair-557/frozen-v4/diagnose_reference_degeneracy_557.py"
PAYLOAD_SHA = "91e60eda6c6ed4a39fb3d9cc226af87743101b22da9c32c5229030ac6cd4cd8a"
PARENT_SHA = "d0d8ba18026d37a1ea64a247e09e4f6e972db0b157d64ad075851ab76c1c4ca4"
ANCHORS = ("coarse_boundary_anchor_rows", "coarse_boundary_anchor_source_nodes",
           "typed_coarse_anchor_output_rows_018", "typed_coarse_anchor_source_ids_018")
FACE_FIELDS = {"source_O_faces", "area_reference_037_source_O_faces",
               "refinement_face_parent_face_row", "refinement_face_parent_barycentric",
               "refinement_face_parent_barycentric_semantics",
               "refinement_face_parent_ancestry_indptr",
               "refinement_face_parent_ancestry_face_rows",
               "refinement_face_parent_ancestry_semantics"}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()

def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}

def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def derive_singleton_barycentric(face_nodes, parent_tri, qptr, qnodes, qw):
    """Exact sparse-Q support-to-parent-triangle barycentric mapping."""
    pos = {int(n): i for i, n in enumerate(parent_tri)}
    out = np.zeros((3, 3), dtype=np.float64)
    for i, node in enumerate(map(int, face_nodes)):
        total = 0.0
        for j in range(int(qptr[node]), int(qptr[node + 1])):
            pnode, w = int(qnodes[j]), float(qw[j])
            if abs(w) <= 1e-15:
                continue
            if pnode not in pos:
                raise ValueError("Q support outside parent triangle")
            out[i, pos[pnode]] += w
            total += w
        if abs(total - 1.0) > 1e-12 or abs(float(out[i].sum()) - 1.0) > 1e-12:
            raise ValueError("Q row does not preserve affine weights")
    return out

def csr(z, name, nrow, ncol):
    ptr = np.asarray(z[name + "_indptr"], dtype=np.int64)
    idx = np.asarray(z[name + "_node_rows"], dtype=np.int64)
    dat = np.asarray(z[name + "_weights"], dtype=np.float64)
    if ptr.shape != (nrow + 1,) or ptr[0] != 0 or ptr[-1] != len(idx) or len(idx) != len(dat) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("invalid CSR " + name)
    if np.any(idx < 0) or np.any(idx >= ncol) or not np.isfinite(dat).all():
        raise ValueError("invalid CSR entries " + name)
    m = sp.csr_matrix((dat, idx, ptr), shape=(nrow, ncol))
    m.sum_duplicates(); m.sort_indices()
    return m

def run(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    search_report_path = SEARCH / "report.json"
    search_report = json.loads(search_report_path.read_text())
    base_state = json.loads(BASE_STATE.read_text())
    parent_state = json.loads(PARENT_STATE.read_text())
    base_path = Path(search_report["input_candidate"])
    candidate_path = Path(search_report["output_candidate"])
    if Path(search_report["input_state"]) != BASE_STATE or sha(BASE_STATE) != search_report["input_state_sha256"]:
        raise ValueError("580 report does not pin supplied 689 baseline state")
    if base_state.get("candidate") != str(base_path) or base_state.get("candidate_sha256") != search_report["input_candidate_sha256"] or sha(base_path) != search_report["input_candidate_sha256"]:
        raise ValueError("580 input does not match completed 689 state")
    if sha(candidate_path) != search_report["output_candidate_sha256"]:
        raise ValueError("580 output hash mismatch")
    if sha(PARENT) != PARENT_SHA or parent_state.get("candidate") != str(PARENT) or parent_state.get("candidate_sha256") != PARENT_SHA:
        raise ValueError("637 parent candidate/state pin mismatch")
    if sha(PAYLOAD) != PAYLOAD_SHA:
        raise ValueError("frozen 568 payload helper changed")
    for path, expected in search_report.get("source_pins", {}).items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError("580 source pin changed: " + path)

    base, cand, parent = load_npz(base_path), load_npz(candidate_path), load_npz(PARENT)
    if set(base) != set(cand):
        raise ValueError("candidate array inventory differs from 689 baseline")
    changed = sorted(k for k in base if not bitwise(base[k], cand[k]))
    if set(changed) - FACE_FIELDS:
        raise ValueError("unexpected changed non-face fields: " + repr(sorted(set(changed) - FACE_FIELDS)))
    for k in ANCHORS:
        if not bitwise(base[k], cand[k]):
            raise ValueError("typed anchor payload changed: " + k)
    if not bitwise(base["points_world_m"], cand["points_world_m"]):
        raise ValueError("topology proposal moved point coordinates")
    if not np.array_equal(cand["source_O_faces"], cand["area_reference_037_source_O_faces"]):
        raise ValueError("candidate current/037 faces differ")
    if not np.array_equal(parent["source_O_faces"], parent["area_reference_037_source_O_faces"]):
        raise ValueError("637 parent current/037 faces differ")

    # Replay the pinned six-operation payload through frozen 568, independently.
    payload = import_module("payload568_refresh637", PAYLOAD)
    replay = {k: np.array(v, copy=True) for k, v in base.items()}
    for op in search_report["accepted_operations"]:
        replay = payload.apply_patch(replay, op)
    if any(not bitwise(replay[k], cand[k]) for k in cand):
        raise ValueError("580 output is not the exact frozen-568 operation replay")

    ncur, npar = len(cand["points_world_m"]), len(parent["points_world_m"])
    q = csr(cand, "immediate_refinement_prolongation", ncur, npar)
    p = csr(parent, "source_prolongation", npar, int(parent["source_prolongation_node_rows"].max()) + 1)
    cp = csr(cand, "source_prolongation", ncur, int(cand["source_prolongation_node_rows"].max()) + 1)
    qp = (q @ p).tocsr(); qp.sum_duplicates(); qp.sort_indices()
    if qp.shape != cp.shape or not np.array_equal(qp.indptr, cp.indptr) or not np.array_equal(qp.indices, cp.indices) or not np.array_equal(qp.data, cp.data):
        raise ValueError("full source P is not exact CSR Q@637-P")
    parent_ref = np.asarray(parent["area_reference_037_points_world_m_f64"], dtype=np.float64)
    cur_ref = np.asarray(cand["area_reference_037_points_world_m_f64"], dtype=np.float64)
    if not np.array_equal(np.asarray(q @ parent_ref), cur_ref):
        raise ValueError("Q does not reproduce propagated 037 reference bitwise")
    for suffix in ("indptr", "node_rows", "weights"):
        if not bitwise(parent["source_prolongation_" + suffix], base["prior_input_parent__source_prolongation_" + suffix]):
            raise ValueError("637 P differs from archived 689 parent P")

    # Use the existing exact-Q affine auditor with the actual 637 parent and its
    # propagated 037 frame; do not use 557's hard-coded historical parent hashes.
    owner = import_module("affine_auditor557", OWNER557)
    canonical = {"source_O_faces": np.asarray(parent["area_reference_037_source_O_faces"]),
                 "points_world_m": parent_ref}
    expected, audit = owner.derive_affine_face_barycentrics(cand, parent, canonical, None)
    if audit["unmapped_singleton_face_count"] != 0:
        raise ValueError("unmapped singleton parent-face rows")
    stale = np.flatnonzero((cand["refinement_face_parent_face_row"] >= 0) &
                           np.any(np.abs(cand["refinement_face_parent_barycentric"] - expected) > 1e-12, axis=(1, 2)))
    if len(stale):
        raise ValueError("637-derived singleton barycentrics are stale: " + repr(stale[:12].tolist()))
    if audit["singleton_parent_face_count"] + audit["multi_parent_face_count"] != len(cand["source_O_faces"]):
        raise ValueError("face lineage counts do not cover full surface")
    # Confirm all 637 face IDs in the emitted CSR are in range.
    aptr = cand["refinement_face_parent_ancestry_indptr"]
    arows = cand["refinement_face_parent_ancestry_face_rows"]
    if aptr.shape != (len(cand["source_O_faces"]) + 1,) or aptr[0] != 0 or aptr[-1] != len(arows) or np.any(aptr[1:] < aptr[:-1]) or np.any(arows < 0) or np.any(arows >= len(parent["source_O_faces"])):
        raise ValueError("invalid or out-of-range 637 face ancestry CSR")

    pins = {str(p): sha(p) for p in (search_report_path, BASE_STATE, base_path, candidate_path, PARENT_STATE, PARENT, PAYLOAD, OWNER557)}
    out.mkdir(parents=True)
    result = {
        "status": "637_PARENT_LINEAGE_VERIFIED_NO_REWRITE_REQUIRED",
        "candidate": str(candidate_path), "candidate_sha256": sha(candidate_path),
        "base_state": str(BASE_STATE), "base_candidate": str(base_path),
        "parent_state": str(PARENT_STATE), "parent_candidate": str(PARENT),
        "changed_fields_from_689": changed,
        "coordinates_and_237_typed_anchors_bitwise_unchanged": True,
        "topology_payload_replay": {"owner": str(PAYLOAD), "owner_sha256": sha(PAYLOAD), "operation_count": len(search_report["accepted_operations"]), "all_arrays_reproduced_bitwise": True},
        "Q_P_reference": {"Q_shape": list(q.shape), "Q_nnz": int(q.nnz), "P_nnz": int(p.nnz), "Q_times_P_exact_csr": True, "Q_times_propagated_037_reference_bitwise": True},
        "face_lineage": {"parent_face_count_637": int(len(parent["source_O_faces"])), "singleton_parent_faces": int(audit["singleton_parent_face_count"]), "multi_parent_faces": int(audit["multi_parent_face_count"]), "unmapped_singleton_faces": 0, "stale_singleton_barycentrics": 0, "max_q_to_037_error_m": audit["Q_to_canonical037_max_abs_point_error_m"], "parent_face_ids_in_range": True, "barycentric_rule": audit["rule"]},
        "pins": pins,
        "limitations": ["Construction-only topology proposal; full 545/130/159 is required.", "No physical, tissue-transfer, native or mechanical qualification."],
        "native_admission": False, "accepted_physical_steps": 0
    }
    (out / "lineage-report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.out)
    print(json.dumps({k:r[k] for k in ("status", "candidate", "candidate_sha256", "face_lineage", "Q_P_reference")}, sort_keys=True), flush=True)
if __name__ == "__main__":
    main()

