#!/usr/bin/env python3
"""Run the real 081 candidate through the existing CPU preparation owners only.

No runtime simulator, GPU, header generation, geometry edits, or physical admission.
The O/I/D arrays are passed unchanged. C is measured in the registered native CSV
frame: O = pointAt(registered CSV) + residual.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E = ROOT / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
PKG = E / "replacement-transfer-ancestry-081-ordered-compact-001"
GEOMETRY = E / "finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz"
PAYLOAD = E / "finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
REGISTERED_CSV = E / "native-pose-checks/connected-hand-registration069i/registered-skin.csv"
MASS_OWNER = E / "replacement-mechanical-061-normal-cone-001/build-report.json"
SHADOW = E / "arm-actuation-preparation-001/barycentric-embedding-owner-001/clean-head-apps-7054-builder-001"
BASE_SHADOW = E / "arm-actuation-preparation-001/barycentric-embedding-owner-001/clean-head-apps"
CPP = E / "replacement-mechanical-081-registered-csv-builder-001/run_081_builder.cpp"
RUNNER = Path(__file__).resolve()
HELPER_PATCH = E / "arm-actuation-preparation-001/barycentric-embedding-owner-001/barycentric-embedding-owner.patch"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def as_contiguous(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype)

def source_constants():
    text = (BASE_SHADOW / "NumiHumanWholeBodySkinReduction.hpp").read_text()
    names = [
        "kWholeBodySkinMinThicknessM", "kWholeBodySkinMaxThicknessM",
        "kWholeBodyFatMinThicknessM", "kWholeBodyFatMaxThicknessM",
    ]
    out = {}
    for name in names:
        m = re.search(rf"inline constexpr double {name}=([^;]+);", text)
        if not m:
            raise ValueError(f"existing owner constant not found in clean shadow: {name}")
        out[name] = float(m.group(1))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for p in [GEOMETRY, PAYLOAD, REGISTERED_CSV, MASS_OWNER, CPP, RUNNER, HELPER_PATCH]:
        if not p.is_file():
            raise FileNotFoundError(p)
    expected_payload_sha = "a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc"
    if sha(PAYLOAD) != expected_payload_sha:
        raise ValueError("the current 069i nhskin payload hash differs from the qualified input")

    # Load the exact compacted 081 candidate and its source ancestry.
    coarse_path = PKG / "coarse-surface.npz"
    layer_path = PKG / "candidate-layers.npz"
    with np.load(coarse_path, allow_pickle=False) as c, np.load(layer_path, allow_pickle=False) as l, np.load(GEOMETRY, allow_pickle=False) as g:
        outer = as_contiguous(l["outer_world"], np.float64)
        inner = as_contiguous(l["inner_world"], np.float64)
        deep = as_contiguous(l["deep_world"], np.float64)
        faces64 = np.asarray(c["faces"], dtype=np.int64)
        parents64 = np.asarray(c["source_face_parent"], dtype=np.int64)
        face_regions = np.asarray(c["face_region"]).astype(str)
        source_ids = as_contiguous(c["source_vertex"], np.int64)
        source_face64 = np.asarray(l["source_embedding_face"], dtype=np.int64)
        beta = as_contiguous(l["source_embedding_barycentric"], np.float64)
        kinds = as_contiguous(c["vertex_kind"], np.uint8)
        skin = as_contiguous(l["skin_thickness_m"], np.float64)
        fat = as_contiguous(l["fat_thickness_m"], np.float64)
        skin_lo = np.asarray(l["skin_lower_bound_m"], dtype=np.float64).copy()
        skin_hi = np.asarray(l["skin_upper_bound_m"], dtype=np.float64).copy()
        fat_lo = np.asarray(l["fat_lower_bound_m"], dtype=np.float64).copy()
        fat_hi = np.asarray(l["fat_upper_bound_m"], dtype=np.float64).copy()
        old_residual = as_contiguous(l["source_embedding_residual_vector_m"], np.float64)
        source_points = np.asarray(g["world"], dtype=np.float64)
        source_faces64 = np.asarray(g["faces"], dtype=np.int64)
        weights64 = np.asarray(g["weights"], dtype=np.float64)
        if outer.shape != inner.shape or outer.shape != deep.shape or outer.shape[1:] != (3,):
            raise ValueError("081 O/I/D shape mismatch")
        n = len(outer)
        if len(kinds) != n or len(source_ids) != n or len(source_face64) != n or beta.shape != (n,3):
            raise ValueError("081 row ancestry does not align with O/I/D")
        if faces64.ndim != 2 or faces64.shape[1] != 3 or parents64.shape != (len(faces64),) or face_regions.shape != (len(faces64),):
            raise ValueError("081 face ancestry is incomplete")
        if source_points.ndim != 2 or source_points.shape[1] != 3 or weights64.shape != (len(source_points),86):
            raise ValueError("detail source geometry or full 86-weight input is incomplete")
        if not np.isfinite(outer).all() or not np.isfinite(inner).all() or not np.isfinite(deep).all():
            raise ValueError("081 O/I/D contains nonfinite points")
        if np.any(faces64 < 0) or np.any(faces64 >= n) or np.any(source_faces64 < 0) or np.any(source_faces64 >= len(source_points)):
            raise ValueError("candidate or source geometry face is out of range")
        source_faces = as_contiguous(source_faces64, np.uint32)
        faces = as_contiguous(faces64, np.uint32)
        parents = as_contiguous(np.where(parents64 < 0, np.iinfo(np.uint32).max, parents64), np.uint32)
        source_face_ids = as_contiguous(np.where(source_face64 < 0, np.iinfo(np.uint32).max, source_face64), np.uint32)

        # Existing donor reference bounds apply only to inherited kind-zero nodes,
        # whose 081 bound arrays intentionally contain [0,+inf]. New rows retain
        # their explicit 081 per-node intervals exactly.
        constants = source_constants()
        inherited = (kinds == 0)
        candidate = ~inherited
        skin_lo[inherited] = constants["kWholeBodySkinMinThicknessM"]
        skin_hi[inherited] = constants["kWholeBodySkinMaxThicknessM"]
        fat_lo[inherited] = constants["kWholeBodyFatMinThicknessM"]
        fat_hi[inherited] = constants["kWholeBodyFatMaxThicknessM"]
        if np.any(skin[inherited] < skin_lo[inherited]) or np.any(skin[inherited] > skin_hi[inherited]):
            raise ValueError("inherited skin thickness violates existing whole-body owner bounds")
        if np.any(fat[inherited] < fat_lo[inherited]) or np.any(fat[inherited] > fat_hi[inherited]):
            raise ValueError("inherited fat thickness violates existing whole-body owner bounds")
        if np.any(skin[candidate] < skin_lo[candidate]) or np.any(skin[candidate] > skin_hi[candidate]):
            raise ValueError("new 081 skin field violates its explicit per-row bounds")
        if np.any(fat[candidate] < fat_lo[candidate]) or np.any(fat[candidate] > fat_hi[candidate]):
            raise ValueError("new 081 fat field violates its explicit per-row bounds")
        skin_lo = as_contiguous(skin_lo, np.float64)
        skin_hi = as_contiguous(skin_hi, np.float64)
        fat_lo = as_contiguous(fat_lo, np.float64)
        fat_hi = as_contiguous(fat_hi, np.float64)

    # Native CSV is the registered point/frame source. Verify stable exact row IDs.
    csv_data = np.genfromtxt(REGISTERED_CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if len(csv_data) != len(source_points):
        raise ValueError("native registered CSV row count differs from source geometry")
    if not np.array_equal(csv_data["vertex"].astype(np.int64), np.arange(len(source_points), dtype=np.int64)):
        raise ValueError("native registered CSV vertex IDs are not exact row indices")
    csv_world = as_contiguous(np.column_stack([csv_data["wx"], csv_data["wy"], csv_data["wz"]]), np.float64)
    csv_donor = np.asarray(csv_data["donor"], dtype=np.uint32)
    csv_local = np.column_stack([csv_data["lx"], csv_data["ly"], csv_data["lz"]]).astype(np.float64)
    csv_matrices = np.column_stack([
        csv_data["m00"], csv_data["m01"], csv_data["m02"],
        csv_data["m10"], csv_data["m11"], csv_data["m12"],
        csv_data["m20"], csv_data["m21"], csv_data["m22"],
    ]).astype(np.float64).reshape((-1,9))
    weights = as_contiguous(weights64, np.float32)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.max(np.abs(weights.astype(np.float64).sum(axis=1)-1.0)) > 1.0e-4:
        raise ValueError("source full 86-body weights fail finite partition checks")
    source_owner_slots = np.argmax(weights64, axis=1)
    slot_to_donor = {}
    for slot, donor in zip(source_owner_slots.tolist(), csv_donor.tolist()):
        previous = slot_to_donor.setdefault(int(slot), int(donor))
        if previous != int(donor):
            raise ValueError(f"source weight slot {slot} maps to multiple native CSV donors")
    if len(slot_to_donor) != len(set(csv_donor.tolist())):
        raise ValueError("native donor IDs and dominant source-weight slots are not one-to-one")
    slot_donor = np.full(86, np.iinfo(np.uint32).max, dtype=np.uint32)
    slot_origin = np.zeros((86,3), dtype=np.float64)
    slot_matrix = np.repeat(np.eye(3,dtype=np.float64).reshape(1,9), 86, axis=0)
    frame_audit = {}
    for slot, donor in slot_to_donor.items():
        rows = np.flatnonzero(csv_donor == donor)
        matrices = csv_matrices[rows]
        if np.max(np.abs(matrices - matrices[0])) > 1.0e-7:
            raise ValueError(f"native CSV donor {donor} has nonconstant world-to-local matrix")
        m = matrices[0].reshape((3,3))
        # CSV local = M * (registered world - donor origin).
        offsets = csv_world[rows] - np.linalg.solve(m, csv_local[rows].T).T
        origin = np.median(offsets, axis=0)
        spread = np.linalg.norm(offsets-origin[None,:], axis=1)
        max_spread = float(spread.max(initial=0.0))
        if max_spread > 1.0e-6:
            raise ValueError(f"native CSV donor {donor} origin reconstruction residual {max_spread}m")
        slot_donor[slot] = donor
        slot_origin[slot] = origin
        slot_matrix[slot] = matrices[0]
        frame_audit[str(slot)] = {"donor_body_index": int(donor), "max_origin_reconstruction_residual_m": max_spread}

    # Build the registered-frame C map; preserve O, and explicitly do not reuse
    # source_embedding_residual_vector_m (which is in the pre-registration frame).
    source_ids = as_contiguous(source_ids, np.int64)
    residual = np.empty_like(outer)
    kind_row_stats = {}
    direct_count = 0
    projected_count = 0
    onehot_count = 0
    for i in range(len(outer)):
        face = int(source_face_ids[i])
        b = beta[i]
        sid = int(source_ids[i])
        if kinds[i] == 0:
            if sid < 0 or sid >= len(csv_world):
                raise ValueError(f"kind-zero row {i} lacks exact registered CSV source ID")
            point = csv_world[sid]
            direct_count += 1
        else:
            if face >= len(source_faces):
                raise ValueError(f"nonzero kind row {i} lacks valid source-face ancestry")
            tri = source_faces[face]
            if not np.isfinite(b).all() or abs(float(b.sum())-1.0) > 1.0e-8 or b.min() < -1.0e-8:
                raise ValueError(f"nonzero kind row {i} has invalid barycentric coordinates")
            point = b @ csv_world[tri]
            is_onehot = any(abs(float(b[k])-1.0) <= 1.0e-8 and all(abs(float(b[j])) <= 1.0e-8 for j in range(3) if j != k) for k in range(3))
            if is_onehot:
                onehot_count += 1
            else:
                projected_count += 1
        residual[i] = outer[i] - point
    registered_residual_norm = np.linalg.norm(residual, axis=1)
    old_new_delta = np.linalg.norm(old_residual[candidate] - residual[candidate], axis=1)
    residual_stats = {
        "max_all_rows_m": float(registered_residual_norm.max()),
        "max_kind_zero_retained_mechanical_delta_m": float(registered_residual_norm[inherited].max()),
        "max_new_row_registered_frame_residual_m": float(registered_residual_norm[candidate].max()),
        "median_new_row_registered_frame_residual_m": float(np.median(registered_residual_norm[candidate])),
        "max_difference_from_old_preg_registration_residual_vector_m": float(old_new_delta.max()),
        "max_old_preg_registration_residual_norm_m_new_rows": float(np.linalg.norm(old_residual[candidate],axis=1).max()),
    }

    mass_report = json.loads(MASS_OWNER.read_text())
    fixed = mass_report["layers"]["fixed_original_mass_owners_kg"]
    debit = as_contiguous([
        fixed["kWholeBodySkinMassDebitKg"],
        fixed["kWholeBodyFatMassDebitKg"],
    ], np.float64)

    output_dir = Path(__file__).resolve().parent
    # Compile the exact clean-HEAD shadow source and test wrapper; no checkout files change.
    lib = out / "lib_081_builder.dylib"
    compile_cmd = [
        "xcrun", "clang++", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-dynamiclib", "-fPIC", "-I"+str(SHADOW), "-I"+str(BASE_SHADOW), "-I"+str(ROOT/"include"),
        "-I"+str(ROOT/"matter/include"), str(CPP), "-o", str(lib),
    ]
    receipt = {
        "scope": "actual 081 candidate CPU preparation/assembly fixture only; no runtime simulation, no GPU, no header generation, no physical admission",
        "base_revision": (E/"arm-actuation-preparation-001/barycentric-embedding-owner-001/clean-head-commit-builder-001.txt").read_text().strip(),
        "clean_shadow_headers": {p.name: sha(p) for p in [SHADOW/"NumiHumanCoarseTissuePreparation.hpp", SHADOW/"NumiHumanSkinFEMIntegration.wholebody.hpp", BASE_SHADOW/"NumiHumanWholeBodySkinReduction.hpp", BASE_SHADOW/"NumiHumanSkinRigidPlacement.hpp"]},
        "inputs": {str(p): sha(p) for p in [coarse_path, layer_path, GEOMETRY, PAYLOAD, REGISTERED_CSV, MASS_OWNER, CPP, RUNNER, HELPER_PATCH]},
        "payload_sha256_expected": "a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc",
        "payload_sha256_actual": sha(PAYLOAD),
        "csv_sha256": sha(REGISTERED_CSV),
        "registration_shift_m": [-0.05,0.20,2.00],
        "residual_definition": "registered CSV frame only: residual[i] = O[i] - pointAt(registered CSV); for direct rows pointAt uses exact source ID, otherwise exact source face/bary. O is unmodified.",
        "excluded_legacy_residual": "candidate-layers.source_embedding_residual_vector_m is pre-registration source-frame metadata; intentionally not reused for current native CSV frame.",
        "candidate_shape": {"vertices": int(len(outer)), "faces": int(len(faces)), "detail_source_vertices": int(len(source_points)), "detail_source_faces": int(len(source_faces))},
        "source_face_parent_mapping": {"rows": int(len(parents64)), "unique_nonnegative_parent_ids": int(len(set(parents64[parents64 >= 0].tolist()))), "missing_parent_rows": int(np.count_nonzero(parents64 < 0)), "missing_parent_region_counts": {str(k): int(np.count_nonzero((parents64 < 0) & (face_regions == k))) for k in sorted(set(face_regions.tolist())) if np.count_nonzero((parents64 < 0) & (face_regions == k))}},
        "row_counts": {"inherited_kind_zero": int(inherited.sum()), "new_nonzero_kind": int(candidate.sum()), "direct_kind_zero": direct_count, "new_onehot_rows": onehot_count, "new_projected_rows": projected_count},
        "source_weight_owner_mapping": {"unique_mapped_slots": int(len(slot_to_donor)), "unique_csv_donors": int(len(set(csv_donor.tolist()))), "dominant_owner_matches_csv_rows": bool(all(int(slot_to_donor[int(s)]) == int(d) for s,d in zip(source_owner_slots,csv_donor))), "frames": frame_audit},
        "registered_frame_residual_stats": residual_stats,
        "thickness_bounds": {"existing_kind_zero_owner_constants_m": constants, "candidate_new_rows_bound_provenance": "candidate-layers.npz explicit per-node lower/upper arrays; no global bound change"},
        "fixed_existing_donor_mass_debit_kg": {"skin": float(debit[0]), "fat": float(debit[1])},
        "compile_command": compile_cmd,
    }
    (out/"invocation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    t0=time.monotonic()
    cp=subprocess.run(compile_cmd, cwd=ROOT, capture_output=True, text=True)
    (out/"compile.stdout").write_text(cp.stdout)
    (out/"compile.stderr").write_text(cp.stderr)
    receipt["compile"]={"exit":cp.returncode,"wall_seconds":time.monotonic()-t0}
    if cp.returncode:
        (out/"result.json").write_text(json.dumps({"status":"COMPILE_FAILED","compile":receipt["compile"]},indent=2)+"\n")
        (out/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps({"output":str(out),"compile":receipt["compile"]},indent=2))
        return 1

    arrays = [
        outer, inner, deep, faces, parents, source_ids, source_face_ids, beta, kinds,
        residual, skin, fat, skin_lo, skin_hi, fat_lo, fat_hi, csv_world, source_faces,
        weights, as_contiguous(slot_donor,np.uint32), as_contiguous(slot_origin,np.float64),
        as_contiguous(slot_matrix,np.float64),
    ]
    ptr_type=ctypes.c_void_p
    ptrs=(ptr_type*len(arrays))(*[int(x.ctypes.data) for x in arrays])
    dims=(ctypes.c_size_t*4)(len(outer),len(faces),len(csv_world),len(source_faces))
    output=ctypes.create_string_buffer(65536)
    library=ctypes.CDLL(str(lib))
    function=library.run081Candidate
    function.argtypes=[ctypes.POINTER(ptr_type),ctypes.POINTER(ctypes.c_size_t),ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_double),ctypes.c_void_p,ctypes.c_size_t]
    function.restype=ctypes.c_int
    t1=time.monotonic()
    code=function(ptrs,dims,sha(GEOMETRY).encode(),sha(PAYLOAD).encode(),debit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),output,len(output))
    elapsed=time.monotonic()-t1
    native_text=output.value.decode("utf-8",errors="replace")
    (out/"native-owner.stdout").write_text(native_text)
    result={
        "status": "BUILDER_PASS_APPEND_PASS" if code==0 else "BUILDER_PASS_APPEND_REJECTED" if code==2 else "BUILDER_REJECTED" if code==1 else "NATIVE_WRAPPER_ERROR",
        "wrapper_return_code": int(code),
        "wall_seconds": elapsed,
        "native_owner_output": native_text,
        "physical_admission": False,
        "known_081_limits": ["081 source outer surface has 45 open head-rim edges; tested O outer has no crossing pairs; derived skin-inner has 6 crossing pairs and fat-deep has 93 crossing pairs; appendFromLayers rejects these layer boundaries.", "081 has 780 source_face_parent=-1 faces, all wrist_transition, plus repeated source parents; this fixture preserves that partial ancestry and does not establish unique detailed face-parent identity or full transfer ancestry."],
        "qualification_statement": "This demonstrates only real registered point/frame and full-86-weight preparation plus the existing append owner rejection. No runtime scene, no FEM solve, no contact solve, no headers, and no physical/biological qualification."
    }
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    receipt["native_owner_run"]={"wrapper_return_code":int(code),"wall_seconds":elapsed}
    receipt["registered_frame_residual_stats"]=residual_stats
    receipt["status"]=result["status"]
    receipt["physical_admission"]=False
    (out/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"output":str(out),"status":result["status"],"wrapper_return_code":int(code),"wall_seconds":elapsed,"registered_frame_residual_stats":residual_stats},indent=2))
    print(native_text,end="")
    # Append rejection is retained as a real result, not converted to a pass.
    return 0 if code==0 else 2 if code==2 else 1

if __name__=="__main__":
    raise SystemExit(main())

