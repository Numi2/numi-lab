from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "numi.connected_remesh_constraint_filter.v1"
EXPECTED = {
    "legacy_base": "73dcee97f00bd4135855e2cf4b2a568eff73a4200836204c92e115ddb6ca741c",
    "remesh_input": "2346d174df8b363bca2200c24589a36cf1ad6a4642a4b73a9a156a48e6826b25",
    "remesh_output": "bc98b04d1035528c82c30aa6e929cd91159c6a17677ffcf9e3690d6866f76e20",
    "flip_report": "57a99f494042e4768e73b29b0348d892a8fdad7767dd016123777a9e047f0e78",
    "self_pairs": "929d1fcb53ed0b4b6ba750aae6669b41725dc9c072f0f27309dd13734ab9dfbb",
    "fixed_bone": "f8ba1b49a6cf11431161065776578b265324a965c1c3205e05ab4c7583301eb9",
}

FACE_ROW_KEYS = (
    "face_kind",
    "parent_coarse_face_row",
    "parent_fine_detail_face_row",
    "parent_source_face_marker",
    "parent_fine_subface_row",
    "refinement_face_parent_face_row",
    "refinement_face_parent_barycentric",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def _faces(z: Any, label: str) -> np.ndarray:
    if "source_O_faces" not in z or "area_reference_037_source_O_faces" not in z:
        raise ValueError(f"{label}: missing current or matched-037 face connectivity")
    f = np.asarray(z["source_O_faces"])
    r = np.asarray(z["area_reference_037_source_O_faces"])
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError(f"{label}: source_O_faces must be integer (F,3)")
    if r.shape != f.shape or r.dtype.kind not in "iu":
        raise ValueError(f"{label}: matched-037 faces shape/type mismatch")
    if not np.array_equal(f, r):
        raise ValueError(f"{label}: current and matched-037 connectivity are not synchronized")
    if np.any(f < 0) or len(f) == 0:
        raise ValueError(f"{label}: invalid face indices")
    return f


def _csr_rows(z: Any, indptr_key: str, data_key: str, nfaces: int, label: str) -> list[np.ndarray]:
    if indptr_key not in z or data_key not in z:
        return []
    ptr = np.asarray(z[indptr_key])
    data = np.asarray(z[data_key])
    if ptr.dtype.kind not in "iu" or ptr.shape != (nfaces + 1,):
        raise ValueError(f"{label}: invalid {indptr_key}")
    if data.dtype.kind not in "iu" or ptr[0] != 0 or ptr[-1] != len(data) or np.any(np.diff(ptr) < 0):
        raise ValueError(f"{label}: invalid {data_key}")
    return [data[int(ptr[i]):int(ptr[i + 1])] for i in range(nfaces)]


def _metadata_changed(a: Any, b: Any, nfaces: int) -> set[int]:
    changed: set[int] = set()
    for key in FACE_ROW_KEYS:
        if key not in a and key not in b:
            continue
        if key not in a or key not in b:
            raise ValueError(f"row-aligned face metadata presence differs for {key}")
        aa = np.asarray(a[key])
        bb = np.asarray(b[key])
        if aa.shape != bb.shape or aa.shape[0] != nfaces:
            raise ValueError(f"row-aligned face metadata shape mismatch for {key}")
        rows = np.flatnonzero(np.any(aa != bb, axis=tuple(range(1, aa.ndim))) if aa.ndim > 1 else aa != bb)
        changed.update(map(int, rows))
    for ptr_key, data_key in (
        ("face_source_ancestry_indptr", "face_source_ancestry_face_rows"),
        ("face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows"),
    ):
        ra = _csr_rows(a, ptr_key, data_key, nfaces, "remesh input")
        rb = _csr_rows(b, ptr_key, data_key, nfaces, "remesh output")
        if bool(ra) != bool(rb):
            raise ValueError(f"face ancestry presence differs for {ptr_key}")
        if ra:
            for i, (x, y) in enumerate(zip(ra, rb)):
                if not np.array_equal(x, y):
                    changed.add(i)
    return changed


def _subset_barriers(arrays: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    required = {
        "skin_face_rows", "bank_body", "stable_bone_id",
        "bank_triangle_stable_id", "fixed_triangles_world_mm",
    }
    if not required.issubset(arrays):
        raise ValueError("fixed-bone file is missing required arrays")
    n = len(arrays["skin_face_rows"])
    if any(len(arrays[k]) != n for k in required):
        raise ValueError("fixed-bone arrays are not row aligned")
    return {k: np.asarray(v)[keep].copy() for k, v in arrays.items()}


def filter_constraint_rows(
    legacy_faces: np.ndarray,
    input_faces: np.ndarray,
    output_faces: np.ndarray,
    reported_changed_rows: list[int] | np.ndarray,
    self_pairs: np.ndarray,
    bone_face_rows: np.ndarray,
    metadata_changed_rows: set[int] | None = None,
) -> dict[str, np.ndarray]:
    legacy_faces = np.asarray(legacy_faces)
    input_faces = np.asarray(input_faces)
    output_faces = np.asarray(output_faces)
    if legacy_faces.shape != input_faces.shape or input_faces.shape != output_faces.shape:
        raise ValueError("face row domains differ; row-index remap is unsafe")
    if not np.array_equal(legacy_faces, input_faces):
        raise ValueError("legacy constraints do not match the pinned remesh input topology")
    nf = len(output_faces)
    changed = np.flatnonzero(np.any(input_faces != output_faces, axis=1))
    reported = np.asarray(reported_changed_rows, dtype=np.int64)
    if reported.ndim != 1 or len(np.unique(reported)) != len(reported):
        raise ValueError("reported changed rows must be a unique vector")
    if np.any(reported < 0) or np.any(reported >= nf) or not np.array_equal(np.sort(reported), changed):
        raise ValueError("flip report changed rows do not match actual topology diff")
    meta = metadata_changed_rows or set()
    unsafe = np.zeros(nf, dtype=bool)
    unsafe[changed] = True
    if meta:
        mr = np.asarray(sorted(meta), dtype=np.int64)
        if np.any(mr < 0) or np.any(mr >= nf):
            raise ValueError("metadata-changed row out of range")
        unsafe[mr] = True
    pairs = np.asarray(self_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("self-pairs must be integer (N,2)")
    if np.any(pairs < 0) or np.any(pairs >= nf):
        raise ValueError("self-pair face row out of range")
    bone = np.asarray(bone_face_rows)
    if bone.ndim != 1 or bone.dtype.kind not in "iu" or np.any(bone < 0) or np.any(bone >= nf):
        raise ValueError("bone face rows must be valid integer face IDs")
    pair_keep = ~np.any(unsafe[pairs], axis=1) if len(pairs) else np.ones(0, dtype=bool)
    bone_keep = ~unsafe[bone]
    return {
        "changed_face_rows": changed.astype(np.int64),
        "metadata_changed_face_rows": np.asarray(sorted(meta), dtype=np.int64),
        "metadata_only_face_rows": np.asarray(sorted(meta - set(map(int, changed))), dtype=np.int64),
        "unsafe_face_rows": np.flatnonzero(unsafe).astype(np.int64),
        "retained_self_pairs": pairs[pair_keep].copy(),
        "stale_self_pairs": pairs[~pair_keep].copy(),
        "retained_bone_mask": bone_keep,
        "stale_bone_mask": ~bone_keep,
    }


def remap(
    legacy_base_path: Path,
    remesh_input_path: Path,
    remesh_output_path: Path,
    flip_report_path: Path,
    self_pairs_path: Path,
    fixed_bone_path: Path,
    output_dir: Path,
    require_pins: bool = True,
) -> dict[str, Any]:
    paths = {
        "legacy_base": legacy_base_path,
        "remesh_input": remesh_input_path,
        "remesh_output": remesh_output_path,
        "flip_report": flip_report_path,
        "self_pairs": self_pairs_path,
        "fixed_bone": fixed_bone_path,
    }
    hashes = {k: sha256_file(v) for k, v in paths.items()}
    if require_pins:
        for key, expected in EXPECTED.items():
            if hashes[key] != expected:
                raise ValueError(f"{key} SHA mismatch: {hashes[key]} != {expected}")
    with np.load(legacy_base_path, allow_pickle=False) as legacy, \
         np.load(remesh_input_path, allow_pickle=False) as source, \
         np.load(remesh_output_path, allow_pickle=False) as target:
        lf = _faces(legacy, "legacy base")
        sf = _faces(source, "remesh input")
        tf = _faces(target, "remesh output")
        if not np.array_equal(lf, sf):
            raise ValueError("legacy constraint base face rows differ from the 359 remesh input")
        if lf.shape != tf.shape:
            raise ValueError("face count changed; this row-preserving filter cannot remap")
        if "points_world_m" not in source or "points_world_m" not in target:
            raise ValueError("missing current point arrays")
        sp = np.asarray(source["points_world_m"])
        tp = np.asarray(target["points_world_m"])
        if sp.dtype != np.float32 or tp.dtype != np.float32 or not np.array_equal(sp, tp):
            raise ValueError("remesh changed current points; legacy geometric barriers cannot be reused")
        if sp.ndim != 2 or sp.shape[1] != 3 or np.any(~np.isfinite(sp)):
            raise ValueError("current points must be finite float32 (N,3)")
        if int(tf.max()) >= len(sp):
            raise ValueError("face node index exceeds current point domain")
        if "area_reference_037_points_world_m_f64" not in source or "area_reference_037_points_world_m_f64" not in target:
            raise ValueError("missing matched-037 reference point arrays")
        sr = np.asarray(source["area_reference_037_points_world_m_f64"])
        tr = np.asarray(target["area_reference_037_points_world_m_f64"])
        if sr.dtype != np.float64 or tr.dtype != np.float64 or not np.array_equal(sr, tr):
            raise ValueError("remesh changed matched-037 reference points")
        if sr.shape[0] != sp.shape[0]:
            raise ValueError("current/reference node domains differ")
        nfaces = len(tf)
        metadata_changed = _metadata_changed(source, target, nfaces)
        with flip_report_path.open("r", encoding="utf-8") as f:
            flip_report = json.load(f)
        if flip_report.get("input_candidate_sha256") != hashes["remesh_input"]:
            raise ValueError("flip report input hash does not match pinned remesh input")
        if flip_report.get("output_candidate_sha256") != hashes["remesh_output"]:
            raise ValueError("flip report output hash does not match pinned remesh output")
        reported = flip_report.get("changed_face_rows")
        if not isinstance(reported, list):
            raise ValueError("flip report has no changed_face_rows list")
    pairs = np.load(self_pairs_path, allow_pickle=False)
    with np.load(fixed_bone_path, allow_pickle=False) as z:
        bone_arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    part = filter_constraint_rows(
        lf, sf, tf, reported, pairs, bone_arrays["skin_face_rows"], metadata_changed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outfiles: dict[str, str] = {}
    for name in ("changed_face_rows", "metadata_changed_face_rows", "metadata_only_face_rows", "unsafe_face_rows"):
        p = output_dir / f"{name.replace('_', '-')}.npy"
        np.save(p, part[name], allow_pickle=False)
        outfiles[p.name] = sha256_file(p)
    for name in ("retained_self_pairs", "stale_self_pairs"):
        p = output_dir / f"{name.replace('_', '-')}.npy"
        np.save(p, part[name], allow_pickle=False)
        outfiles[p.name] = sha256_file(p)
    for label, mask_key in (("retained", "retained_bone_mask"), ("stale", "stale_bone_mask")):
        arrays = _subset_barriers(bone_arrays, part[mask_key])
        p = output_dir / f"{label}-fixed-bone-barriers.npz"
        np.savez_compressed(p, **arrays)
        outfiles[p.name] = sha256_file(p)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PARTIAL_LEGACY_CONSTRAINT_FILTER",
        "native_admission": False,
        "constraint_use": "identity-preserving seed only; retained rows are not a complete current audit",
        "required_followup": [
            "rerun complete exact self-surface audit on the 359 candidate",
            "rerun all 185 typed-bank skin-bone audits on the 359 candidate",
            "combine fresh witnesses with retained identities only after auditing current geometry",
        ],
        "inputs": {
            k: {"path": str(p), "sha256": hashes[k]} for k, p in paths.items()
        },
        "topology": {
            "face_count": int(len(tf)),
            "legacy_base_points_bitwise_equal_to_remesh_input": bool(np.array_equal(
                np.load(legacy_base_path, allow_pickle=False)["points_world_m"], sp
            )),
            "retained_constraints_are_identity_seeds_not_reused_geometry_coefficients": True,
            "legacy_and_remesh_input_connectivity_sha256": array_sha256(lf),
            "remesh_output_connectivity_sha256": array_sha256(tf),
            "flip_report_changed_face_rows": int(len(part["changed_face_rows"])),
            "provenance_metadata_changed_face_row_count": int(len(part["metadata_changed_face_rows"])),
            "provenance_only_changed_face_rows": int(len(part["metadata_only_face_rows"])),
            "unsafe_face_row_count": int(len(part["unsafe_face_rows"])),
            "unsafe_face_rows_sha256": array_sha256(part["unsafe_face_rows"]),
            "changed_face_rows_sha256": array_sha256(part["changed_face_rows"]),
            "current_points_bitwise_unchanged_across_remesh": True,
            "matched_037_reference_points_bitwise_unchanged_across_remesh": True,
            "matched_037_connectivity_synchronized": True,
            "face_row_identity_semantics": "same row index only; no stable-ID or nearest-face inference",
        },
        "constraints": {
            "self_pairs_input": int(len(pairs)),
            "self_pairs_retained_identity_only": int(len(part["retained_self_pairs"])),
            "self_pairs_stale_for_fresh_audit": int(len(part["stale_self_pairs"])),
            "fixed_bone_rows_input": int(len(bone_arrays["skin_face_rows"])),
            "fixed_bone_rows_retained_identity_only": int(np.count_nonzero(part["retained_bone_mask"])),
            "fixed_bone_rows_stale_for_fresh_audit": int(np.count_nonzero(part["stale_bone_mask"])),
            "fixed_bone_triangle_geometry_copied_verbatim_mm": True,
        },
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": digest}
            for name, digest in outfiles.items()
        },
        "no_357_358_witness_rows_reused": True,
    }
    report_path = output_dir / "remap-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed row-identity filter for legacy face constraints after 359 flips.")
    ap.add_argument("--legacy-base", type=Path, required=True)
    ap.add_argument("--remesh-input", type=Path, required=True)
    ap.add_argument("--remesh-output", type=Path, required=True)
    ap.add_argument("--flip-report", type=Path, required=True)
    ap.add_argument("--self-pairs", type=Path, required=True)
    ap.add_argument("--fixed-bone", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--skip-pins", action="store_true", help="test-only; production invocation must use frozen SHA pins")
    args = ap.parse_args()
    report = remap(args.legacy_base, args.remesh_input, args.remesh_output, args.flip_report,
                   args.self_pairs, args.fixed_bone, args.output_dir, require_pins=not args.skip_pins)
    print(json.dumps({
        "status": report["status"],
        "unsafe_face_row_count": report["topology"]["unsafe_face_row_count"],
        "self_pairs_retained": report["constraints"]["self_pairs_retained_identity_only"],
        "self_pairs_stale": report["constraints"]["self_pairs_stale_for_fresh_audit"],
        "bone_rows_retained": report["constraints"]["fixed_bone_rows_retained_identity_only"],
        "bone_rows_stale": report["constraints"]["fixed_bone_rows_stale_for_fresh_audit"],
        "report": str(args.output_dir / "remap-report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

