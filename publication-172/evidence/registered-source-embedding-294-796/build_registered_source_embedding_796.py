#!/usr/bin/env python3
"""Pinned, inert source-rest embedding proof for the 796 exact crossing pairs."""
from __future__ import annotations
import csv, hashlib, json, struct, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OUT = ROOT / "D/registered-source-embedding-294-796"
R0P = ROOT / "full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
CANDP = ROOT / "D/surface-clearance-546/connected-batch-619/run-005-from878/round-002/restore-620/candidate.npz"
STATEP = ROOT / "D/surface-clearance-546/connected-batch-619/run-005-from878/round-002/final/state.json"
AUDITP = ROOT / "D/surface-clearance-546/connected-batch-619/run-005-from878/round-002/restored-check/audit-159/report.json"
P1P = ROOT / "publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
SKINP = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
BONEP = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.nhbones")
RIGIDP = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
CSV = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv")
CPP = Path("/Users/n/numi-human-aortic-contact-opt-20261019/apps/numilab_human_myosim_visual_probe.mm")
EMBED_HPP = Path("/Users/n/numi-human-aortic-contact-opt-20261019/apps/NumiHumanSkinFEMIntegration.wholebody.hpp")
CONSUMER_HPP = Path("/Users/n/numi-human-aortic-contact-opt-20261019/apps/NumiHumanWholeBodySkinEmbeddingConsumer.hpp")
PY = "/Users/n/numi-human-prep-venv-20261005/bin/python"
EXPECTED = {
    str(R0P): "70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31",
    str(CANDP): "1a79cf4051e61e2c92124958b90098e6e6bd509f0424118fa85cad5bce69fd5a",
    str(AUDITP): "3c690616ca00789846fdf174bca072a2127abe9787a94343f5b9e062503fd8d7",
    str(P1P): "21f1869b418a2eca1e5a0522059c8f9cc6f8afe1a270ccc7c7d99c9c19fdd2a4",
    str(SKINP): "d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04",
    str(CSV): "71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14",
}
RESID_TOL_M = 2.0e-8
BARY_TOL = 2.0e-6

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def csr_from(z, name, rows, cols):
    return csr_matrix((z[name + "_weights"], z[name + "_node_rows"], z[name + "_indptr"]), shape=(rows, cols))

def barycentric(x, tri):
    a, b, c = tri
    u, v, w = b - a, c - a, x - a
    g00, g01, g11 = float(u @ u), float(u @ v), float(v @ v)
    det = g00 * g11 - g01 * g01
    if not np.isfinite(det) or det <= 1e-28:
        return None
    uv = np.linalg.solve(np.array([[g00, g01], [g01, g11]]), np.array([u @ w, v @ w]))
    beta = np.array([1.0 - uv.sum(), uv[0], uv[1]], dtype=np.float64)
    return beta, float(np.linalg.norm(beta @ tri - x))

def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    for path, expected in EXPECTED.items():
        actual = sha(Path(path))
        if actual != expected:
            raise SystemExit(f"pinned input hash mismatch: {path}: {actual} != {expected}")
    state = json.loads(STATEP.read_text())
    audit = json.loads(AUDITP.read_text())
    if state["candidate"] != str(CANDP) or state["candidate_sha256"] != EXPECTED[str(CANDP)]:
        raise SystemExit("state candidate path/hash mismatch")
    if audit["candidate_sha256"] != EXPECTED[str(CANDP)] or not audit["complete"] or audit["bank_count"] != 185:
        raise SystemExit("159 receipt is not a complete candidate-bound 185-bank audit")
    if state["native_admission"] or state["accepted_physical_steps"] != 0:
        raise SystemExit("unexpected admission/physical-step state")
    R0, C, P1 = [np.load(p, allow_pickle=False) for p in (R0P, CANDP, P1P)]
    pair_rows, pair_banks = [], []
    for bank in audit["banks"]:
        r = bank.get("result")
        if not isinstance(r, dict):
            continue
        for sf, bf in r["triangle_pairs"]:
            pair_rows.append((int(sf), int(bf)))
            pair_banks.append((int(bank["body"]), int(bank["stable_bone_id"])))
    pairs = np.asarray(pair_rows, dtype=np.int64).reshape(-1, 2)
    banks = np.asarray(pair_banks, dtype=np.int32).reshape(-1, 2)
    if len(pairs) != 796:
        raise SystemExit(f"expected pinned 796-pair report, got {len(pairs)}")
    cross_faces = np.unique(pairs[:, 0])
    cross_nodes = np.unique(C["source_O_faces"][cross_faces].reshape(-1)).astype(np.int64)
    if len(cross_faces) != 199 or len(cross_nodes) != 232:
        raise SystemExit("crossing face/node coverage differs from expected pinned receipt")
    # NHSKIN/NHBONES/NHRIGID registration header contract.
    skin_bytes = SKINP.read_bytes()
    skin_head = struct.unpack_from("<8s5I32s", skin_bytes, 0)
    magic, abi, bind_count, vert_count, index_count, fingerprint, skin_source_sha = skin_head
    if magic != b"NHSKIN1\0" or abi != 5 or bind_count != 86 or vert_count != 61950:
        raise SystemExit("NHSKIN header does not match registered full-weight 294 payload")
    binding_offset = 60
    binding_body_ids = np.array([struct.unpack_from("<I", skin_bytes, binding_offset + i * 36)[0] for i in range(bind_count)], dtype=np.uint32)
    vertex_offset = binding_offset + bind_count * 36
    index_offset = vertex_offset + vert_count * 56
    skin_faces = np.frombuffer(skin_bytes, dtype="<u4", count=index_count, offset=index_offset).reshape(-1, 3).astype(np.int64)
    weight_offset = index_offset + index_count * 4
    weights = np.frombuffer(skin_bytes, dtype="<f4", count=vert_count * bind_count, offset=weight_offset).reshape(vert_count, bind_count)
    expected_size = weight_offset + vert_count * bind_count * 4
    if len(skin_bytes) != expected_size:
        raise SystemExit("NHSKIN full-weight payload byte size mismatch")
    bone_head = struct.unpack_from("<8s5I32s", BONEP.read_bytes(), 0)
    rigid_head = struct.unpack_from("<8s10I32s", RIGIDP.read_bytes(), 0)
    if bone_head[0] != b"NHBONES1" or bone_head[2] != 185:
        raise SystemExit("NHBONES registration header mismatch")
    if rigid_head[0] != b"NHRIGID2":
        raise SystemExit("NHRIGID source header mismatch")
    bone_fingerprint = int(bone_head[5])
    bone_source_sha = bone_head[6].hex()
    rigid_source_sha = rigid_head[-1].hex()
    if fingerprint != bone_fingerprint or skin_source_sha.hex() != bone_source_sha or rigid_source_sha != bone_source_sha:
        raise SystemExit("skin/bone/rigid registration fingerprint or source digest mismatch")
    if len(np.unique(binding_body_ids)) != bind_count:
        raise SystemExit("registered skin binding body IDs are not unique")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise SystemExit("invalid full-body source weights")
    weight_sum_error = np.max(np.abs(weights.sum(axis=1, dtype=np.float64) - 1.0))
    if weight_sum_error > 1e-4:
        raise SystemExit(f"full-body source weights do not normalize: {weight_sum_error}")
    # P1's detail row domain and exact registered-rest transfer table.
    ids = P1["detail_source_vertex_id"].astype(np.int64)
    src_faces = P1["source_surface_faces_detail_rows"].astype(np.int64)
    src_xyz = P1["detail_baseline_world_m"].astype(np.float64)
    if len(ids) != vert_count or not np.array_equal(ids, np.arange(vert_count)):
        raise SystemExit("P1 detail source IDs are not the NHSKIN vertex domain")
    if not np.array_equal(src_faces, skin_faces):
        raise SystemExit("P1 detail face rows do not exactly equal NHSKIN face indices")
    csv_rows = np.empty((vert_count, 3), dtype=np.float64)
    csv_ids = np.empty(vert_count, dtype=np.int64)
    csv_detail = np.empty(vert_count, dtype=np.int64)
    with CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["detail_row"])
            csv_detail[i] = i
            csv_ids[i] = int(row["detail_source_vertex_id"])
            csv_rows[i] = [float(row["detail_rest_world_x_m"]), float(row["detail_rest_world_y_m"]), float(row["detail_rest_world_z_m"])]
    csv_delta = np.abs(csv_rows - src_xyz)
    if not np.array_equal(csv_detail, np.arange(vert_count)) or not np.array_equal(csv_ids, ids):
        raise SystemExit("accepted transfer rows do not cover the P1 detail identity domain")
    if np.max(csv_delta) > 1e-12:
        raise SystemExit("P1 rest-world coordinates differ from accepted transfer rows")
    # Root R0 -> P1 source-rest points: direct detail ID for kinds 1/2;
    # unique source_status==1 coarse representative for kind 0; preserve missing roots.
    id_to_row = {int(v): i for i, v in enumerate(ids)}
    rep = P1["representative_coarse_row"].astype(np.int64)
    status = P1["source_status"] == 1
    values, counts = np.unique(rep[status & (rep >= 0)], return_counts=True)
    rep_to_row = {int(v): int(np.flatnonzero(status & (rep == v))[0]) for v, n in zip(values, counts) if n == 1}
    root_kind = R0["node_kind"].astype(np.int64)
    root_source = R0["node_source_row"].astype(np.int64)
    root_p1_row = np.full(len(root_kind), -1, dtype=np.int64)
    root_class = np.zeros(len(root_kind), dtype=np.int8) # 0 unmapped, 1 direct ID, 2 unique representative
    for i, (kind, source) in enumerate(zip(root_kind, root_source)):
        if kind in (1, 2) and int(source) in id_to_row:
            root_p1_row[i] = id_to_row[int(source)]
            root_class[i] = 1
        elif kind == 0 and int(source) in rep_to_row:
            root_p1_row[i] = rep_to_row[int(source)]
            root_class[i] = 2
    root_xyz = np.full((len(root_kind), 3), np.nan, dtype=np.float64)
    mapped = root_p1_row >= 0
    root_xyz[mapped] = src_xyz[root_p1_row[mapped]]
    # Current-to-R0 Q/P/CSR chain, exact stored CSR only.
    q1 = csr_from(C, "source_prolongation", len(C["points_world_m"]), len(C["prior_input_006__source_prolongation_indptr"]) - 1)
    q2 = csr_from(C, "prior_input_006__source_prolongation", q1.shape[1], len(C["prior_run028__source_prolongation_indptr"]) - 1)
    q3 = csr_from(C, "prior_run028__source_prolongation", q2.shape[1], len(root_kind))
    if q1.shape[1] != q2.shape[0] or q2.shape[1] != q3.shape[0] or q3.shape[1] != len(root_kind):
        raise SystemExit("Q/P source-prolongation dimensions do not compose")
    Q = q1 @ q2 @ q3
    qcross = Q[cross_nodes]
    missing_mass = np.asarray(qcross[:, ~mapped].sum(axis=1)).reshape(-1)
    qsum = np.asarray(qcross.sum(axis=1)).reshape(-1)
    if np.max(missing_mass) > 1e-14 or np.max(np.abs(qsum - 1.0)) > 1e-12:
        raise SystemExit("crossing-node Q/P stencil is incomplete or not affine")
    source_rest = np.asarray(qcross @ np.nan_to_num(root_xyz, nan=0.0))
    # Fine-detail candidate faces are the explicit current face ancestry union
    # over crossing faces. Test all, never choose an arbitrary parent.
    fptr = C["face_fine_detail_ancestry_indptr"]
    frows = C["face_fine_detail_ancestry_rows"].astype(np.int64)
    candidates = {int(n): set() for n in cross_nodes}
    for face in cross_faces:
        lo, hi = int(fptr[face]), int(fptr[face + 1])
        face_ancestry = {int(x) for x in frows[lo:hi] if 0 <= int(x) < len(src_faces)}
        for node in C["source_O_faces"][face]:
            candidates[int(node)].update(face_ancestry)
    node_lookup = {int(n): i for i, n in enumerate(cross_nodes)}
    all_sf, all_beta, all_resid, all_vertex_rows, all_weights = [], [], [], [], []
    candidate_indptr = [0]
    node_class = []
    ambiguity_records = []
    for node in cross_nodes:
        qi = node_lookup[int(node)]
        valid_faces = []
        for sf in sorted(candidates[int(node)]):
            face_vertices = src_faces[sf]
            result = barycentric(source_rest[qi], src_xyz[face_vertices])
            if result is None:
                continue
            beta, residual = result
            if residual <= RESID_TOL_M and beta.min() >= -BARY_TOL and beta.max() <= 1 + BARY_TOL:
                valid_faces.append((sf, beta, residual))
        if not valid_faces:
            raise SystemExit(f"no ancestry-bound P1 source-face embedding for crossing node {node}")
        node_class.append(1 if len(valid_faces) == 1 else 2)
        if len(valid_faces) == 2:
            shared = sorted(set(src_faces[valid_faces[0][0]].tolist()) & set(src_faces[valid_faces[1][0]].tolist()))
            if len(shared) != 2:
                raise SystemExit(f"two-face ambiguity is not a shared source edge for node {node}: {shared}")
            ambiguity_records.append({"node_row": int(node), "source_face_rows": [int(x[0]) for x in valid_faces], "shared_source_vertex_rows": shared})
        for sf, beta, residual in valid_faces:
            face_vertices = src_faces[sf]
            w86 = beta @ weights[face_vertices].astype(np.float64)
            if abs(float(w86.sum()) - 1.0) > 2e-4:
                raise SystemExit(f"interpolated 86-weight row fails for node {node}, source face {sf}")
            all_sf.append(sf); all_beta.append(beta); all_resid.append(residual)
            all_vertex_rows.append(face_vertices); all_weights.append(w86)
        candidate_indptr.append(len(all_sf))
    current = C["points_world_m"][cross_nodes].astype(np.float64)
    delta_mm = np.linalg.norm(current - source_rest, axis=1) * 1000.0
    node_counts = {str(i): int(sum(c == i for c in node_class)) for i in (1, 2)}
    by_bank = []
    pair_by_bank = {}
    for bank in sorted({tuple(x) for x in banks.tolist()}):
        mask = np.all(banks == bank, axis=1)
        fs = np.unique(pairs[mask, 0])
        ns = np.unique(C["source_O_faces"][fs].reshape(-1))
        d = delta_mm[[node_lookup[int(n)] for n in ns]]
        by_bank.append({
            "body": bank[0], "stable_bone_id": bank[1], "pair_count": int(mask.sum()),
            "skin_face_count": int(len(fs)), "crossing_node_count": int(len(ns)),
            "unique_source_face_nodes": int(sum(len(candidates[int(n)]) > 0 and node_class[node_lookup[int(n)]] == 1 for n in ns)),
            "shared_edge_ambiguous_nodes": int(sum(node_class[node_lookup[int(n)]] == 2 for n in ns)),
            "source_rest_delta_mm_p50": float(np.percentile(d, 50)),
            "source_rest_delta_mm_max": float(np.max(d)),
        })
    # Compact Q support for these crossing nodes records exact R0 ancestry without expanding a generic provenance table.
    csr = qcross.tocsr()
    report = {
        "status": "INERT_REGISTERED_SOURCE_REST_EMBEDDING_DIAGNOSTIC",
        "geometry_or_native_admission": False,
        "candidate_selected": False,
        "physical_steps": 0,
        "remote_command": f"ssh numi {PY} {Path(__file__).resolve()}",
        "inputs": {str(p): pin(p) for p in [R0P, CANDP, STATEP, AUDITP, P1P, SKINP, BONEP, RIGIDP, CSV, CPP, EMBED_HPP, CONSUMER_HPP]},
        "pinned_expected_hashes_checked": EXPECTED,
        "candidate_state": {"candidate_sha256": state["candidate_sha256"], "accepted_physical_steps": state["accepted_physical_steps"], "native_admission": state["native_admission"], "continuation_mode": state.get("continuation_mode")},
        "exact159_receipt": {"status": audit["status"], "candidate_sha256": audit["candidate_sha256"], "complete": audit["complete"], "bank_count": audit["bank_count"], "pair_count": int(len(pairs)), "crossing_skin_face_count": int(len(cross_faces)), "crossing_node_count": int(len(cross_nodes)), "active_bank_count": int(len(by_bank)), "elapsed_seconds": audit.get("elapsed_seconds")},
        "registered_source": {
            "skin_payload_magic": magic.decode("ascii").rstrip("\\0"), "skin_payload_abi": int(abi),
            "skin_vertex_count": int(vert_count), "skin_face_count": int(index_count // 3), "full_binding_count": int(bind_count),
            "skin_registration_fingerprint_hex": f"0x{fingerprint:08x}",
            "bone_count": int(bone_head[2]), "bone_registration_fingerprint_hex": f"0x{bone_fingerprint:08x}",
            "skin_bone_rigid_source_sha256_equal": True, "rigid_engine_body_count": int(rigid_head[4]),
            "unique_binding_body_ids": int(len(np.unique(binding_body_ids))),
            "max_full86_weight_row_sum_error": float(weight_sum_error),
            "max_interpolated_86_weight_row_sum_error": float(max(abs(float((np.asarray(all_weights).sum(axis=1)) .max())-1.0), 0.0)) if False else float(np.max(np.abs(np.asarray(all_weights).sum(axis=1)-1.0))),
            "p1_detail_source_ids_equal_skin_vertex_rows": True, "p1_source_face_indices_equal_nhskin_indices": True,
            "accepted_transfer_csv_rows": int(len(csv_detail)), "p1_rest_world_max_abs_difference_from_csv_m": float(np.max(csv_delta)),
            "owner_functions": [
                {"path": str(CPP), "symbol": "loadSkin", "line": 2898, "contract": "checks NHBK registration fingerprint, source digest, full-weight ABI, binding and geometry dimensions"},
                {"path": str(CPP), "symbol": "skin payload admission call", "line": 30731, "contract": "passes bonePayload.header.reserved0 as expected registration fingerprint"},
                {"path": str(EMBED_HPP), "symbol": "evaluateRegisteredSkinSourceEmbedding", "line": 40, "contract": "validates source face+bary, evaluates registered positions and interpolates the complete 86-body weight vector"},
                {"path": str(CONSUMER_HPP), "symbol": "registered skin embedding consumer", "line": 158, "contract": "existing consumer for registered source embedding"},
            ],
        },
        "r0_to_p1_rest_mapping": {
            "semantics": "kind 1/2 node_source_row exact detail_source_vertex_id; kind 0 node_source_row resolves only through a unique P1 representative_coarse_row with source_status==1; kind 5 and other unmatched roots remain unmapped",
            "r0_root_node_count": int(len(root_kind)),
            "direct_detail_id_rows": int(np.sum(root_class == 1)),
            "unique_coarse_representative_rows": int(np.sum(root_class == 2)),
            "unmapped_root_rows": int(np.sum(root_class == 0)),
            "root_source_map_class_codes": {"0": "unmapped", "1": "direct_detail_id", "2": "unique_source_status_1_coarse_representative"},
        },
        "current_crossing_embedding": {
            "q_p_chain": ["source_prolongation", "prior_input_006__source_prolongation", "prior_run028__source_prolongation"],
            "q_p_shape_current_to_r0": [int(Q.shape[0]), int(Q.shape[1])],
            "q_p_nnz": int(Q.nnz), "crossing_q_support_nnz": int(qcross.nnz),
            "crossing_q_row_sum_min": float(qsum.min()), "crossing_q_row_sum_max": float(qsum.max()),
            "crossing_q_unmapped_root_mass_max": float(missing_mass.max()),
            "mapped_crossing_nodes": int(np.sum(np.asarray([len(candidates[int(n)]) > 0 for n in cross_nodes]))),
            "source_face_mapping_classes": {"unique_face_barycentric": int(node_counts["1"]), "shared_edge_ambiguous": int(node_counts["2"]), "unmapped": int(len(cross_nodes) - sum(node_counts.values()))},
            "source_face_candidate_count_total": int(len(all_sf)),
            "max_source_face_barycentric_reconstruction_residual_m": float(max(all_resid)),
            "barycentric_tolerance": BARY_TOL, "residual_tolerance_m": RESID_TOL_M,
            "source_rest_delta_to_current_candidate_mm_p50": float(np.percentile(delta_mm, 50)),
            "source_rest_delta_to_current_candidate_mm_p90": float(np.percentile(delta_mm, 90)),
            "source_rest_delta_to_current_candidate_mm_max": float(np.max(delta_mm)),
            "source_rest_delta_over_12mm_count_diagnostic_only": int(np.sum(delta_mm > 12.0)),
            "shared_edge_ambiguities": ambiguity_records,
        },
        "by_bank": by_bank,
        "limits": [
            "Source-rest coordinates are P1 registered-rest coordinates; this does not evaluate a new posed anatomical state.",
            "Face+bary mappings are ancestry-bound to exact P1/294 source faces; shared-edge alternatives are retained, not selected.",
            "No point was moved and no geometry/self/containment/cell/soft-tissue physical admission was performed by this diagnostic.",
            "The exact 159 receipt is a surface crossing report only; it is not native or physical qualification.",
        ],
        "elapsed_seconds": time.monotonic() - start,
    }
    npzout = OUT / "crossing-source-rest.npz"
    np.savez_compressed(
        npzout,
        exact159_pair_skin_bone_rows=pairs,
        exact159_pair_bank_body_stable_id=banks,
        crossing_skin_face_rows=cross_faces.astype(np.int64),
        crossing_node_rows=cross_nodes,
        crossing_node_current_world_m_f32=C["points_world_m"][cross_nodes],
        crossing_node_source_rest_world_m=source_rest,
        crossing_node_source_rest_delta_mm=delta_mm,
        crossing_node_source_mapping_class=np.asarray(node_class, dtype=np.int8),
        crossing_node_q_r0_indptr=csr.indptr.astype(np.int64),
        crossing_node_q_r0_rows=csr.indices.astype(np.int64),
        crossing_node_q_r0_weights=csr.data.astype(np.float64),
        r0_root_source_class=root_class,
        r0_root_p1_detail_row=root_p1_row,
        source_candidate_indptr=np.asarray(candidate_indptr, dtype=np.int64),
        source_candidate_face_rows=np.asarray(all_sf, dtype=np.int64),
        source_candidate_barycentric=np.asarray(all_beta, dtype=np.float64),
        source_candidate_residual_m=np.asarray(all_resid, dtype=np.float64),
        source_candidate_vertex_rows=np.asarray(all_vertex_rows, dtype=np.int64),
        source_candidate_detail_source_vertex_ids=ids[np.asarray(all_vertex_rows, dtype=np.int64)],
        source_candidate_full86_weights=np.asarray(all_weights, dtype=np.float64),
    )
    report["output_npz"] = pin(npzout)
    report["script_sha256"] = sha(Path(__file__))
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    execution = {
        "status": "PASS_INERT_MAPPING_AND_PIN_ASSERTIONS",
        "command": [PY, str(Path(__file__).resolve())],
        "script_sha256": sha(Path(__file__)),
        "report": pin(report_path),
        "npz": pin(npzout),
        "return_code": 0,
        "native_or_geometry_run": False,
    }
    (OUT / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": execution["status"], "report": str(report_path), "npz": str(npzout), "report_sha256": sha(report_path), "npz_sha256": sha(npzout), "unique_nodes": node_counts["1"], "ambiguous_nodes": node_counts["2"], "pairs": len(pairs)}, sort_keys=True))

if __name__ == "__main__":
    main()
