#!/usr/bin/env python3
"""Refresh one bank in the existing 191 registered-tissue cache format.

Only the selected stable surface's exact191 vertex/face arrays and matching
inventory row are re-emitted. The cache's other arrays and inventory rows are
preserved byte-for-byte/value-for-value. This is a source-cache rebuild, not
mechanical or physical admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class RefreshError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: str | Path, expected: str, name: str) -> str:
    expected = str(expected).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RefreshError(f"{name} SHA-256 must be 64 lowercase hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise RefreshError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _load_owner(path: Path, expected_sha: str):
    _pin(path, expected_sha, "canonical NHTISS owner")
    spec = importlib.util.spec_from_file_location("canonical_nhtiss_cache_refresh_owner", path)
    if spec is None or spec.loader is None:
        raise RefreshError("cannot import canonical NHTISS owner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("decode_nhtiss", "emit_current", "load_pose_map", "load_exact_191_helpers"):
        if not callable(getattr(module, name, None)):
            raise RefreshError(f"canonical NHTISS owner lacks {name}")
    return module


def _face_array(payload: dict[str, Any], record: np.ndarray) -> np.ndarray:
    _fb, _bc, fv, vc, fi, ic, stable_id, _layer = map(int, record)
    faces = payload["indices"][fi:fi + ic].reshape(-1, 3)
    if np.any(faces < fv) or np.any(faces >= fv + vc):
        raise RefreshError(f"stable surface {stable_id} has indices outside its vertex range")
    return (faces.astype(np.uint64) - np.uint64(fv)).astype(np.uint32)


def _array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def refresh_one_bank(*, owner: Any, baseline_raw: bytes, candidate_raw: bytes,
                     stable_id: int, pose_data: dict[str, Any], inventory: dict[str, Any],
                     cache_arrays: dict[str, np.ndarray], baseline_sha256: str,
                     candidate_sha256: str, pose_sha256: str, inventory_sha256: str,
                     cache_sha256: str, emitter_sha256: str, rotate: Any,
                     bodyworld: Any) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    baseline = owner.decode_nhtiss(baseline_raw)
    candidate = owner.decode_nhtiss(candidate_raw)
    if (baseline["magic"], baseline["abi"], baseline["nr"], baseline["nb"], baseline["nv"],
        baseline["registration"], baseline["source_id"]) != (
        candidate["magic"], candidate["abi"], candidate["nr"], candidate["nb"], candidate["nv"],
        candidate["registration"], candidate["source_id"]):
        raise RefreshError("baseline and candidate NHTISS source domains differ")
    if not np.array_equal(baseline["bindings"], candidate["bindings"]):
        raise RefreshError("topology candidate unexpectedly changed binding bytes")
    if not np.array_equal(baseline["vertices"], candidate["vertices"]):
        raise RefreshError("topology candidate unexpectedly changed vertex, normal, or weight bytes")
    if not np.array_equal(baseline["records"][:, [0, 1, 2, 3, 6, 7]],
                          candidate["records"][:, [0, 1, 2, 3, 6, 7]]):
        raise RefreshError("candidate changed stable IDs or non-index record domains")
    ids = baseline["records"][:, 6].astype(np.int64)
    hits = np.flatnonzero(ids == int(stable_id))
    if len(hits) != 1:
        raise RefreshError(f"stable surface {stable_id} must occur once")
    target_row = int(hits[0])
    baseline_record = baseline["records"][target_row]
    candidate_record = candidate["records"][target_row]
    bfb, bbc, bfv, bvc, _bfi, _bic, _bsid, _blayer = map(int, baseline_record)
    cfb, cbc, cfv, cvc, _cfi, _cic, _csid, _clayer = map(int, candidate_record)
    if (bfb, bbc, bfv, bvc) != (cfb, cbc, cfv, cvc):
        raise RefreshError("target stable surface changed binding or vertex range")
    for row, sid in enumerate(ids):
        if int(sid) == int(stable_id):
            continue
        if not np.array_equal(_face_array(baseline, baseline["records"][row]),
                              _face_array(candidate, candidate["records"][row])):
            raise RefreshError(f"candidate changed unrelated NHTISS topology at stable ID {sid}")

    pose_map = owner.load_pose_map(pose_data, baseline["registration"])
    rotate, bodyworld = rotate, bodyworld
    baseline_current, _ = owner.emit_current(baseline, baseline_record, pose_map, rotate, bodyworld)
    candidate_current, _ = owner.emit_current(candidate, candidate_record, pose_map, rotate, bodyworld)
    bank_rows = [row for row in inventory.get("banks", [])
                 if row.get("kind") == "soft_tissue" and int(row.get("stable_id", -1)) == int(stable_id)]
    if len(bank_rows) != 1:
        raise RefreshError(f"inventory must contain exactly one soft-tissue bank {stable_id}")
    bank = bank_rows[0]
    key = str(bank.get("key", ""))
    vertex_key, face_key = key + "_vertices", key + "_faces"
    if vertex_key not in cache_arrays or face_key not in cache_arrays:
        raise RefreshError(f"cache lacks target bank arrays for stable ID {stable_id}")
    cached_vertices, cached_faces = cache_arrays[vertex_key], cache_arrays[face_key]
    if cached_vertices.dtype != np.float32 or cached_vertices.shape != baseline_current.shape:
        raise RefreshError("baseline cache target vertices have the wrong shape or dtype")
    baseline_faces = _face_array(baseline, baseline_record)
    candidate_faces = _face_array(candidate, candidate_record)
    if cached_faces.dtype != np.uint32 or not np.array_equal(cached_faces, baseline_faces):
        raise RefreshError("baseline cached target face sequence is not exact NHTISS topology")
    if not np.array_equal(cached_vertices, baseline_current):
        delta = float(np.linalg.norm((cached_vertices - baseline_current).astype(np.float64), axis=1).max())
        raise RefreshError(f"baseline current cache is not exact191 parity ({delta} m)")
    if not np.array_equal(candidate_current, cached_vertices):
        delta = float(np.linalg.norm((candidate_current - cached_vertices).astype(np.float64), axis=1).max())
        raise RefreshError(f"topology-only candidate moved target current vertices ({delta} m)")
    if len(candidate_faces) != int(candidate_record[5]) // 3:
        raise RefreshError("candidate target face count differs from its record")

    updated_arrays = {name: np.asarray(value).copy() for name, value in cache_arrays.items()}
    updated_arrays[vertex_key] = candidate_current.astype(np.float32, copy=True)
    updated_arrays[face_key] = candidate_faces.astype(np.uint32, copy=True)
    preserved_hashes: dict[str, str] = {}
    for name, old_array in cache_arrays.items():
        new_array = updated_arrays[name]
        if name == face_key:
            continue
        if old_array.dtype != new_array.dtype or old_array.shape != new_array.shape or _array_bytes(old_array) != _array_bytes(new_array):
            raise RefreshError(f"non-target cache array changed: {name}")
        preserved_hashes[name] = hashlib.sha256(_array_bytes(old_array)).hexdigest()
    if len(preserved_hashes) != len(cache_arrays) - 1:
        raise RefreshError("expected only the target face array to differ")

    old_inventory_banks = inventory.get("banks", [])
    updated_inventory = json.loads(json.dumps(inventory))
    target_index = next(i for i, row in enumerate(updated_inventory["banks"])
                        if row.get("kind") == "soft_tissue" and int(row.get("stable_id", -1)) == int(stable_id))
    old_target_inventory_row = json.loads(json.dumps(updated_inventory["banks"][target_index]))
    used = candidate["vertices"][cfv:cfv + cvc]
    active = used["w"] > np.float32(2e-6)
    body_ids = np.unique(candidate["bindings"][cfb:cfb + cbc]["body"][used["binding"][active]])
    updated_inventory["banks"][target_index]["vertices"] = cvc
    updated_inventory["banks"][target_index]["triangles"] = len(candidate_faces)
    if list(map(int, body_ids)) != bank.get("body_indices"):
        raise RefreshError("topology-only candidate changed the target inventory body identities")
    if candidate_current.min(axis=0).astype(float).tolist() != bank.get("aabb_min_m") or \
       candidate_current.max(axis=0).astype(float).tolist() != bank.get("aabb_max_m"):
        raise RefreshError("topology-only candidate changed the target inventory AABB")
    for index, row in enumerate(old_inventory_banks):
        if index != target_index and updated_inventory["banks"][index] != row:
            raise RefreshError(f"unrelated inventory bank row changed at index {index}")
    input_matches = [row for row in updated_inventory.get("inputs", [])
                     if row.get("kind") == "soft_tissue" and
                     str(row.get("sha256", "")).lower() == baseline_sha256.lower()]
    if len(input_matches) != 1 or int(input_matches[0].get("record_count", -1)) != baseline["nr"]:
        raise RefreshError("inventory does not uniquely pin the baseline soft-tissue payload")
    input_matches[0]["path"] = "__CANDIDATE_PAYLOAD_PATH__"
    input_matches[0]["sha256"] = candidate_sha256
    if updated_inventory.get("pose_sha256", "").lower() != pose_sha256.lower():
        raise RefreshError("inventory pose hash does not match the pinned current pose")
    if updated_inventory.get("cache_sha256", "").lower() != cache_sha256.lower():
        raise RefreshError("inventory cache hash does not match the pinned cache")
    expected_cache_keys = {str(row["key"]) + suffix for row in old_inventory_banks
                           for suffix in ("_vertices", "_faces")}
    if (len(updated_inventory["banks"]) != len(old_inventory_banks) or
        set(cache_arrays) != expected_cache_keys or len(cache_arrays) != 2 * len(old_inventory_banks)):
        raise RefreshError("unexpected baseline inventory/cache cardinality")
    report = {
        "schema": "numi.human.registered-tissue-cache.single-topology-refresh.v1",
        "source_pins": {
            "baseline_payload_sha256": baseline_sha256,
            "candidate_payload_sha256": candidate_sha256,
            "candidate_manifest_sha256": None,
            "pose_sha256": pose_sha256,
            "baseline_inventory_sha256": inventory_sha256,
            "baseline_cache_sha256": cache_sha256,
            "exact_191_emitter_sha256": emitter_sha256,
        },
        "stable_id": int(stable_id),
        "bank_key": key,
        "arrays": {
            "array_count": len(cache_arrays),
            "preserved_exact_count": len(preserved_hashes),
            "preserved_array_sha256": preserved_hashes,
            "updated_target_face_array": face_key,
            "target_vertex_array_reemitted_equal": True,
            "old_target_face_count": len(baseline_faces),
            "new_target_face_count": len(candidate_faces),
            "candidate_target_vertices_sha256": hashlib.sha256(candidate_current.tobytes()).hexdigest(),
            "candidate_target_faces_sha256": hashlib.sha256(candidate_faces.tobytes()).hexdigest(),
        },
        "inventory": {
            "bank_count": len(old_inventory_banks),
            "target_row_before": old_target_inventory_row,
            "target_row_after": updated_inventory["banks"][target_index],
            "unrelated_bank_rows_exact": True,
            "soft_tissue_input_repointed_to_candidate": True,
            "pose_path_and_hash_preserved": True,
            "anatomy_input_preserved": True,
        },
        "stale_downstream_maps": [
            "Existing 009/010 clipped face-row arrays reference the baseline stable134 face numbering and are not reusable.",
            "Any mechanical/material face map indexed by the old stable134 face rows is invalid until rebuilt.",
        ],
        "admission": {"geometry_candidate_cache_only": True, "mechanical_map_valid": False,
                      "physical_admission": False, "native_admission": False},
    }
    return updated_arrays, updated_inventory, report


def _stage_new(path: Path, writer) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RefreshError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(prefix=".cache-refresh-", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    return temp


def _publish_staged(staged: list[tuple[Path, Path]]) -> None:
    if any(final.exists() for final, _temp in staged):
        raise RefreshError("an output appeared during validation; refusing to overwrite")
    published: list[Path] = []
    try:
        for final, temp in staged:
            os.replace(temp, final)
            published.append(final)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for _final, temp in staged:
            temp.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("baseline", "baseline ABI-5 NHTISS source"),
        ("candidate", "topology-replaced ABI-5 NHTISS source"),
        ("candidate-manifest", "manifest that pins the candidate payload"),
        ("pose", "current identity snapshot with all_bodies"),
        ("inventory", "existing 191 cache inventory"),
        ("cache", "existing 191 cache NPZ"),
        ("emitter", "existing exact 191 emitter source"),
        ("owner", "canonical NHTISS source owner"),
    ):
        parser.add_argument(f"--{name}", required=True, help=help_text)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--stable-id", required=True, type=int)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-inventory", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = [Path(args.output_cache), Path(args.output_inventory), Path(args.report)]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("cache, inventory, and report output paths must differ")
    if any(path.exists() for path in outputs):
        parser.error("refusing to overwrite existing output")
    staged_paths: list[Path] = []
    try:
        names = ("baseline", "candidate", "candidate_manifest", "pose", "inventory", "cache", "emitter", "owner")
        actual = {name: _pin(getattr(args, name), getattr(args, name + "_sha256"), name)
                  for name in names}
        candidate_manifest = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))
        if str(candidate_manifest.get("payload", {}).get("sha256", "")).lower() != actual["candidate"]:
            raise RefreshError("candidate manifest payload hash does not match candidate input")
        if int(candidate_manifest.get("payload", {}).get("payload_abi", -1)) != 5:
            raise RefreshError("candidate manifest does not declare NHTISS ABI 5")
        owner = _load_owner(Path(args.owner), actual["owner"])
        baseline_raw = Path(args.baseline).read_bytes()
        candidate_raw = Path(args.candidate).read_bytes()
        pose_data = json.loads(Path(args.pose).read_text(encoding="utf-8"))
        inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        with np.load(args.cache, allow_pickle=False) as archive:
            cache_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        pose_map = owner.load_pose_map(pose_data, owner.decode_nhtiss(baseline_raw)["registration"])
        rotate, bodyworld = owner.load_exact_191_helpers(args.emitter, actual["emitter"], pose_map)
        arrays, updated_inventory, report = refresh_one_bank(
            owner=owner, baseline_raw=baseline_raw, candidate_raw=candidate_raw,
            stable_id=args.stable_id, pose_data=pose_data, inventory=inventory,
            cache_arrays=cache_arrays, baseline_sha256=actual["baseline"],
            candidate_sha256=actual["candidate"], pose_sha256=actual["pose"],
            inventory_sha256=actual["inventory"], cache_sha256=actual["cache"],
            emitter_sha256=actual["emitter"], rotate=rotate, bodyworld=bodyworld)
        updated_inventory["inputs"] = [dict(row) for row in updated_inventory["inputs"]]
        for row in updated_inventory["inputs"]:
            if row.get("kind") == "soft_tissue" and row.get("sha256") == actual["candidate"]:
                row["path"] = str(Path(args.candidate).resolve())
        # A sentinel left by the pure function prevents an unpinned path from
        # leaking into the serialized inventory.
        if any(row.get("path") == "__CANDIDATE_PAYLOAD_PATH__" for row in updated_inventory["inputs"]):
            for row in updated_inventory["inputs"]:
                if row.get("path") == "__CANDIDATE_PAYLOAD_PATH__":
                    row["path"] = str(Path(args.candidate).resolve())
        updated_inventory["cache_path"] = str(outputs[0].resolve())
        cache_temp = _stage_new(outputs[0], lambda stream: np.savez_compressed(stream, **arrays))
        staged_paths.append(cache_temp)
        report["source_pins"]["candidate_manifest_sha256"] = actual["candidate_manifest"]
        report["outputs"] = {"cache_path": str(outputs[0].resolve()), "cache_sha256": sha256_file(cache_temp),
                             "inventory_path": str(outputs[1].resolve())}
        updated_inventory["cache_sha256"] = report["outputs"]["cache_sha256"]
        # Read back the serialized cache and prove exact array preservation.
        with np.load(cache_temp, allow_pickle=False) as readback:
            if set(readback.files) != set(arrays):
                raise RefreshError("serialized cache array key set changed")
            for key in arrays:
                got = np.asarray(readback[key])
                if got.dtype != arrays[key].dtype or got.shape != arrays[key].shape or _array_bytes(got) != _array_bytes(arrays[key]):
                    raise RefreshError(f"serialized cache array readback changed: {key}")
        inv_raw = (json.dumps(updated_inventory, indent=2) + "\n").encode("utf-8")
        report["outputs"]["inventory_sha256"] = hashlib.sha256(inv_raw).hexdigest()
        inventory_temp = _stage_new(outputs[1], lambda stream: stream.write(inv_raw))
        staged_paths.append(inventory_temp)
        report_raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        report_temp = _stage_new(outputs[2], lambda stream: stream.write(report_raw))
        staged_paths.append(report_temp)
        _publish_staged([(outputs[0], cache_temp), (outputs[1], inventory_temp), (outputs[2], report_temp)])
        staged_paths.clear()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (RefreshError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        print(f"refresh_registered_tissue_cache: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
