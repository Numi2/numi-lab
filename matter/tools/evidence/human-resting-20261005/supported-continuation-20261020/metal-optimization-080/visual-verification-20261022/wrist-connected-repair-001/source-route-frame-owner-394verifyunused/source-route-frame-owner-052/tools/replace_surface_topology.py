#!/usr/bin/env python3
"""Rewrite one NHTISS ABI-5 surface topology from a pinned construction patch.

All source vertex, normal, route-weight, binding, and stable-surface identities
remain unchanged. The resulting payload is a topology-only diagnostic candidate:
its old mechanical/material face map is invalid and must be rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class TopologyError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(path: str | Path, expected: str, what: str) -> str:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower()):
        raise TopologyError(f"{what} SHA-256 pin must be 64 hex characters")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise TopologyError(f"{what} hash mismatch: expected {expected.lower()}, got {actual}")
    return actual


def load_canonical_owner(path: str | Path, expected_sha: str):
    require_sha(path, expected_sha, "canonical NHTISS owner")
    spec = importlib.util.spec_from_file_location("canonical_nhtiss_owner_396", path)
    if spec is None or spec.loader is None:
        raise TopologyError("cannot load pinned canonical NHTISS owner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "decode_nhtiss", None)) or not callable(getattr(mod, "validate_nhtiss", None)):
        raise TopologyError("pinned canonical owner lacks ABI decoder/validator")
    return mod


def _directed_boundary(faces: np.ndarray, label: str) -> set[tuple[int, int]]:
    directed = Counter()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        directed[(int(a), int(b))] += 1
        directed[(int(b), int(c))] += 1
        directed[(int(c), int(a))] += 1
    undirected: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for edge, count in directed.items():
        key = tuple(sorted(edge))
        undirected.setdefault(key, []).extend([edge] * count)
    boundary: set[tuple[int, int]] = set()
    for edge, occurrences in undirected.items():
        if len(occurrences) == 1:
            boundary.add(occurrences[0])
        elif len(occurrences) == 2:
            if occurrences[0] != occurrences[1][::-1]:
                raise TopologyError(f"{label}: internal edge {edge} is not oppositely oriented")
        else:
            raise TopologyError(f"{label}: edge {edge} has incidence {len(occurrences)}")
    return boundary


def replace_surface_topology(raw: bytes, owner: Any, *, stable_id: int,
                              patch_face_rows: np.ndarray, fan_faces_local: np.ndarray,
                              boundary_cycle: np.ndarray, ancestry_offsets: np.ndarray,
                              ancestry_face_rows: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    src = owner.decode_nhtiss(raw)
    records = src["records"]
    hits = np.flatnonzero(records[:, 6] == np.uint32(stable_id))
    if len(hits) != 1:
        raise TopologyError(f"stable surface {stable_id} must occur exactly once")
    target_record_index = int(hits[0])
    target = records[target_record_index]
    fb, bc, fv, vc, fi, ic, sid, layer = map(int, target)
    if sid != stable_id or ic % 3:
        raise TopologyError("target record identity/index count is malformed")
    old_global = src["indices"][fi:fi + ic].reshape(-1, 3)
    if np.any(old_global < fv) or np.any(old_global >= fv + vc):
        raise TopologyError("target surface indices are not in its declared vertex range")
    old_local = (old_global.astype(np.int64) - fv).astype(np.uint32)
    patch_rows = np.asarray(patch_face_rows, dtype=np.int64).reshape(-1)
    fan = np.asarray(fan_faces_local, dtype=np.int64).reshape(-1, 3)
    cycle = np.asarray(boundary_cycle, dtype=np.int64).reshape(-1)
    offsets = np.asarray(ancestry_offsets, dtype=np.int64).reshape(-1)
    ancestry = np.asarray(ancestry_face_rows, dtype=np.int64).reshape(-1)
    if not len(patch_rows) or len(set(map(int, patch_rows))) != len(patch_rows):
        raise TopologyError("patch source rows must be a nonempty unique list")
    if np.any(patch_rows < 0) or np.any(patch_rows >= len(old_local)):
        raise TopologyError("patch source face row is outside target surface")
    if fan.shape[0] != len(cycle) or fan.shape[0] < 3 or len(set(map(int, cycle))) != len(cycle):
        raise TopologyError("fan face count/boundary cycle is malformed")
    if np.any(fan < 0) or np.any(fan >= vc) or np.any(fan[:, 0] == fan[:, 1]) or np.any(fan[:, 1] == fan[:, 2]) or np.any(fan[:, 2] == fan[:, 0]):
        raise TopologyError("fan contains invalid local vertex indices")
    if len({tuple(sorted(map(int, f))) for f in fan}) != len(fan):
        raise TopologyError("fan contains duplicate faces")
    if offsets.shape != (len(fan) + 1,) or offsets[0] != 0 or offsets[-1] != len(ancestry) or np.any(np.diff(offsets) <= 0):
        raise TopologyError("construction ancestry CSR offsets are malformed")
    expected_ancestry = sorted(map(int, patch_rows))
    for row in range(len(fan)):
        if sorted(map(int, ancestry[offsets[row]:offsets[row + 1]])) != expected_ancestry:
            raise TopologyError(f"fan face {row} lacks the full declared construction ancestry")

    old_patch = old_local[patch_rows]
    old_boundary = _directed_boundary(old_patch, "old patch")
    cycle_edges = {(int(cycle[i]), int(cycle[(i + 1) % len(cycle)])) for i in range(len(cycle))}
    if old_boundary != cycle_edges:
        raise TopologyError("fixed boundary cycle does not exactly match old directed patch boundary")
    new_boundary = _directed_boundary(fan, "replacement fan")
    if new_boundary != cycle_edges:
        raise TopologyError("replacement fan does not exactly preserve directed patch boundary")

    patch_set = set(map(int, patch_rows))
    insert_at = min(patch_set)
    new_local_rows: list[np.ndarray] = []
    inserted = False
    for row in range(len(old_local)):
        if row == insert_at:
            new_local_rows.extend(fan.astype(np.uint32))
            inserted = True
        if row not in patch_set:
            new_local_rows.append(old_local[row].copy())
    if not inserted:
        raise TopologyError("patch insertion point was not reached")
    new_local = np.asarray(new_local_rows, dtype=np.uint32).reshape(-1, 3)
    if len(new_local) != len(old_local) - len(patch_set) + len(fan):
        raise TopologyError("replacement face count accounting failed")

    # Every record must own one contiguous, non-overlapping slice of the index
    # array. Repacking the slices preserves each unaffected record's exact
    # triangle order and values, while updating subsequent offsets.
    ordered = sorted(range(len(records)), key=lambda i: int(records[i, 4]))
    cursor = 0
    new_records = records.copy()
    new_slices: list[np.ndarray] = []
    target_local_unmodified: list[np.ndarray] = []
    for ri in ordered:
        row = records[ri]
        rfi, ric = int(row[4]), int(row[5])
        if rfi != cursor:
            raise TopologyError("source NHTISS record index slices are not contiguous in record order")
        old_slice = src["indices"][rfi:rfi + ric]
        if ri == target_record_index:
            replacement = (new_local.astype(np.uint64) + np.uint64(fv)).astype(np.uint32).reshape(-1)
            new_slices.append(replacement)
            new_records[ri, 5] = len(replacement)
            target_local_unmodified = [old_local[j].copy() for j in range(len(old_local)) if j not in patch_set]
        else:
            new_slices.append(old_slice.copy())
        new_records[ri, 4] = sum(len(part) for part in new_slices[:-1])
        cursor += ric
    if cursor != int(src["ni"]):
        raise TopologyError("source records do not cover the full NHTISS index array")
    new_indices = np.concatenate(new_slices).astype("<u4", copy=False)
    if len(new_indices) != int(src["ni"]) + 3 * (len(fan) - len(patch_set)):
        raise TopologyError("global index count accounting failed")

    # Only target ic and index starts following its slice may change. All
    # vertices, normals, weights, bindings, records, and stable IDs remain.
    new_header = owner.HEADER.pack(src["magic"], src["abi"], src["nr"], src["nb"],
                                   src["nv"], len(new_indices), src["registration"],
                                   src["source_id"])
    out_raw = (new_header + new_records.astype("<u4", copy=False).tobytes() +
               src["bindings"].tobytes() + src["vertices"].tobytes() +
               new_indices.tobytes())
    out = owner.decode_nhtiss(out_raw)
    if not np.array_equal(out["bindings"], src["bindings"]):
        raise TopologyError("serialized rewrite changed binding bytes")
    if not np.array_equal(out["vertices"], src["vertices"]):
        raise TopologyError("serialized rewrite changed vertex position/normal/route bytes")
    if not np.array_equal(out["records"][:, [0, 1, 2, 3, 6, 7]], records[:, [0, 1, 2, 3, 6, 7]]):
        raise TopologyError("serialized rewrite changed non-index record metadata")
    # Prove all unselected record triangle arrays are exactly unchanged.
    unchanged_records = []
    for ri in range(len(records)):
        if ri == target_record_index:
            continue
        old_row, new_row = records[ri], out["records"][ri]
        old_idx = src["indices"][int(old_row[4]):int(old_row[4] + old_row[5])]
        new_idx = out["indices"][int(new_row[4]):int(new_row[4] + new_row[5])]
        if not np.array_equal(old_idx, new_idx):
            raise TopologyError(f"unselected stable surface {int(old_row[6])} topology changed")
        unchanged_records.append(int(old_row[6]))
    out_target = out["records"][target_record_index]
    out_local = (out["indices"][int(out_target[4]):int(out_target[4] + out_target[5])]
                 .reshape(-1, 3).astype(np.int64) - fv)
    fan_set = {tuple(map(int, face)) for face in fan}
    # Exact list-level check: target nonpatch rows preserve source sequence;
    # every new face appears once.
    fan_observed = [tuple(map(int, face)) for face in out_local if tuple(map(int, face)) in fan_set]
    if Counter(fan_observed) != Counter(tuple(map(int, face)) for face in fan):
        raise TopologyError("target readback does not contain each fan face exactly once")
    nonfan_observed = [face for face in out_local if tuple(map(int, face)) not in fan_set]
    expected_nonfan = [face.astype(np.int64) for face in target_local_unmodified]
    if len(nonfan_observed) != len(expected_nonfan) or any(not np.array_equal(a, b) for a, b in zip(nonfan_observed, expected_nonfan)):
        raise TopologyError("target nonpatch faces changed order or connectivity")

    report = {
        "schema": "numi.human.nhtiss.single-surface-topology-replacement.v1",
        "status": "candidate_only_mechanical_face_map_invalid",
        "stable_id": int(stable_id),
        "existing_abi": "NHTISS4 ABI 5",
        "source_index_count": int(src["ni"]),
        "candidate_index_count": int(out["ni"]),
        "source_surface_face_count": int(ic // 3),
        "candidate_surface_face_count": int(int(out_target[5]) // 3),
        "removed_source_face_rows": sorted(patch_set),
        "replacement_face_count": int(len(fan)),
        "fan_source_vertex_triangles_local": fan.astype(int).tolist(),
        "fixed_boundary_cycle_local": cycle.astype(int).tolist(),
        "construction_ancestry_offsets": offsets.astype(int).tolist(),
        "construction_ancestry_source_face_rows": ancestry.astype(int).tolist(),
        "construction_ancestry_semantics": "every fan face lists all 18 replaced source rows as construction ancestry; not an exact overlap, area, volume, material, or FEM ancestry certificate",
        "unselected_stable_ids_exact_topology_preserved": sorted(unchanged_records),
        "all_vertex_position_normal_route_weight_bytes_preserved": True,
        "all_binding_body_transform_bytes_preserved": True,
        "old_mechanical_face_map": "INVALIDATED for stable surface 134; rebuild from candidate topology before any mechanical/material/FEM integration",
        "physical_admission": False,
    }
    return out_raw, report


def _load_sidecar(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = ("stable_id", "source_patch_face_rows", "fan_source_vertex_triangles",
                    "fixed_boundary_cycle", "construction_ancestry_offsets",
                    "construction_ancestry_source_face_rows")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise TopologyError(f"topology sidecar missing {missing}")
        return {key: np.asarray(z[key]).copy() for key in required}


def _derived_manifest(source_doc: dict[str, Any], payload_sha: str, payload_bytes: int,
                      candidate_index_count: int, report: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(source_doc))
    out["payload"]["sha256"] = payload_sha
    out["payload"]["bytes"] = int(payload_bytes)
    out["payload"]["index_count"] = int(candidate_index_count)
    out["status"] = "diagnostic_topology_candidate_only"
    out["evidence_boundary"] = (
        "Topology-only NHTISS candidate with unchanged vertex, route, and binding bytes. "
        "Not a material map, FEM map, or physical admission."
    )
    out.setdefault("source", {})["topology_candidate"] = {
        "schema": report["schema"],
        "stable_id": report["stable_id"],
        "removed_source_face_rows": report["removed_source_face_rows"],
        "replacement_face_count": report["replacement_face_count"],
        "construction_ancestry_semantics": report["construction_ancestry_semantics"],
        "mechanical_face_map_status": "invalidated_rebuild_required",
        "physical_admission": False,
    }
    return out


def _atomic_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TopologyError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(prefix=".topology-candidate-", dir=path.parent, delete=False) as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    if path.exists():
        tmp.unlink(missing_ok=True)
        raise TopologyError(f"refusing to overwrite {path}")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--sidecar-sha256", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--canonical-owner", required=True)
    ap.add_argument("--canonical-owner-sha256", required=True)
    ap.add_argument("--stable-id", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--derived-manifest", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)
    outputs = [Path(args.output), Path(args.derived_manifest), Path(args.report)]
    if len({p.resolve() for p in outputs}) != len(outputs):
        ap.error("output, derived manifest, and report paths must differ")
    if any(p.exists() for p in outputs):
        ap.error("refusing to overwrite an existing output")
    try:
        pins = {
            "source": require_sha(args.source, args.source_sha256, "source NHTISS"),
            "sidecar": require_sha(args.sidecar, args.sidecar_sha256, "topology sidecar"),
            "manifest": require_sha(args.manifest, args.manifest_sha256, "source manifest"),
            "canonical_owner": require_sha(args.canonical_owner, args.canonical_owner_sha256, "canonical owner"),
        }
        owner = load_canonical_owner(args.canonical_owner, pins["canonical_owner"])
        raw = Path(args.source).read_bytes()
        side = _load_sidecar(args.sidecar)
        stable_id = int(side["stable_id"].reshape(-1)[0])
        if stable_id != args.stable_id:
            raise TopologyError("requested stable ID differs from topology sidecar")
        candidate, report = replace_surface_topology(
            raw, owner, stable_id=stable_id,
            patch_face_rows=side["source_patch_face_rows"],
            fan_faces_local=side["fan_source_vertex_triangles"],
            boundary_cycle=side["fixed_boundary_cycle"],
            ancestry_offsets=side["construction_ancestry_offsets"],
            ancestry_face_rows=side["construction_ancestry_source_face_rows"])
        base_doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        output_sha = sha256_bytes(candidate)
        report["inputs"] = {
            "source_nhtiss_sha256": pins["source"],
            "topology_sidecar_sha256": pins["sidecar"],
            "source_manifest_sha256": pins["manifest"],
            "canonical_owner_sha256": pins["canonical_owner"],
        }
        report["inputs"]["topology_materializer_sha256"] = sha256_file(__file__)
        report["output"] = {"path": str(Path(args.output)), "sha256": output_sha, "bytes": len(candidate)}
        derived = _derived_manifest(base_doc, output_sha, len(candidate),
                                    int(owner.decode_nhtiss(candidate)["ni"]), report)
        derived["payload"]["file"] = Path(args.output).name
        derived_raw = (json.dumps(derived, indent=2, sort_keys=True) + "\n").encode()
        report["output"]["derived_manifest_path"] = str(Path(args.derived_manifest))
        report["output"]["derived_manifest_sha256"] = sha256_bytes(derived_raw)
        report_raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        for p in outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise TopologyError(f"refusing to overwrite {p}")
        # Candidate/report/derived metadata are staged only after canonical
        # decode and all topology/preservation checks have passed.
        _atomic_new(outputs[0], candidate)
        try:
            _atomic_new(outputs[1], derived_raw)
            _atomic_new(outputs[2], report_raw)
        except Exception:
            for p in outputs:
                p.unlink(missing_ok=True)
            raise
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (TopologyError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"replace_surface_topology: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

