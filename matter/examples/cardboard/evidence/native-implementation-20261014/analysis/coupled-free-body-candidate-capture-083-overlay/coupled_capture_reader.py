#!/usr/bin/env python3
"""Hash-checked reader and raw-output comparator for coupled-capture v1."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "numi.matter.coupled_newton_capture.v1"
CAPTURE_POINT = "after_fgmres_measure_and_human_direction_map_before_armijo_trial"


class CaptureError(ValueError):
    pass


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


class Capture:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        try:
            self.manifest = json.loads(self.manifest_path.read_text())
        except Exception as exc:
            raise CaptureError(f"cannot parse manifest: {exc}") from exc
        m = self.manifest
        if m.get("schema") != SCHEMA:
            raise CaptureError("unsupported or missing capture schema")
        if m.get("byte_order") != "little_endian":
            raise CaptureError("capture is not explicitly little endian")
        if m.get("capture_point") != CAPTURE_POINT:
            raise CaptureError("unexpected capture point")
        if m.get("capture_kind") != "raw_transaction_candidate_not_checkpoint":
            raise CaptureError("payload is not marked as a raw candidate")
        payload = m.get("payload")
        if not isinstance(payload, dict):
            raise CaptureError("payload descriptor missing")
        expected_bytes = payload.get("byte_count")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise CaptureError("invalid payload byte count")
        if expected_bytes > 32 * 1024 * 1024:
            raise CaptureError("payload exceeds the 32 MiB runtime capture cap")
        named_path = Path(str(payload.get("file", "")))
        candidates = [named_path] if named_path.is_absolute() else []
        candidates.append(self.manifest_path.parent / named_path.name)
        self.payload_path = next((p for p in candidates if p.is_file()), None)
        if self.payload_path is None:
            raise CaptureError(f"payload file not found: {payload.get('file')}")
        self.payload = self.payload_path.read_bytes()
        if len(self.payload) != expected_bytes:
            raise CaptureError("payload size does not match manifest")
        if sha256_bytes(self.payload) != payload.get("sha256"):
            raise CaptureError("payload SHA-256 mismatch")
        arrays = m.get("arrays")
        max_arrays = m.get("max_arrays")
        if not isinstance(arrays, list) or not isinstance(max_arrays, int):
            raise CaptureError("array table or array bound missing")
        if max_arrays != 192 or len(arrays) > max_arrays or \
                m.get("max_payload_bytes") != 32 * 1024 * 1024 or \
                m.get("array_count") != len(arrays):
            raise CaptureError("array table exceeds declared safety bound")
        self.arrays: dict[str, dict[str, Any]] = {}
        spans: list[tuple[int, int, str]] = []
        for entry in arrays:
            if not isinstance(entry, dict):
                raise CaptureError("array table entry is not an object")
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in self.arrays:
                raise CaptureError(f"invalid or duplicate array name: {name!r}")
            count = entry.get("count")
            itemsize = entry.get("element_bytes")
            offset = entry.get("offset")
            nbytes = entry.get("byte_count")
            if not all(isinstance(v, int) and v >= 0 for v in
                       (count, itemsize, offset, nbytes)) or itemsize == 0:
                raise CaptureError(f"invalid extent for array {name}")
            if count * itemsize != nbytes:
                raise CaptureError(f"count/element size mismatch for {name}")
            end = offset + nbytes
            if end > len(self.payload):
                raise CaptureError(f"array {name} extends past payload")
            view = memoryview(self.payload)[offset:end]
            if sha256_bytes(view) != entry.get("sha256"):
                raise CaptureError(f"SHA-256 mismatch for array {name}")
            spans.append((offset, end, name))
            self.arrays[name] = entry
        spans.sort()
        cursor = 0
        for start, end, name in spans:
            if start != cursor:
                raise CaptureError(f"array payload gap or overlap before {name}")
            cursor = end
        if cursor != len(self.payload):
            raise CaptureError("unclaimed trailing bytes in payload")

    def raw(self, name: str) -> memoryview:
        try:
            entry = self.arrays[name]
        except KeyError as exc:
            raise CaptureError(f"capture has no array {name!r}") from exc
        start = entry["offset"]
        return memoryview(self.payload)[start:start + entry["byte_count"]]

    def numeric(self, name: str) -> np.ndarray:
        entry = self.arrays.get(name)
        if entry is None:
            raise CaptureError(f"capture has no array {name!r}")
        kind = entry.get("dtype")
        if kind == "byte":
            dtype = np.dtype("u1")
            width = 1
        else:
            match = re.fullmatch(r"(float32|uint32)(?:x([1-9][0-9]*))?", str(kind))
            if match is None:
                raise CaptureError(f"array {name} is opaque ({kind}), not numeric")
            dtype = np.dtype("<f4" if match.group(1) == "float32" else "<u4")
            width = int(match.group(2) or "1")
        if entry["element_bytes"] != dtype.itemsize * width:
            raise CaptureError(f"dtype width mismatch for {name}")
        result = np.frombuffer(self.raw(name), dtype=dtype)
        if width != 1:
            result = result.reshape(entry["count"], width)
        return result

    def f64(self, name: str) -> np.ndarray:
        """Widen stored little-endian FP32 words to FP64 without decimal parsing."""
        values = self.numeric(name)
        if values.dtype.kind != "f":
            raise CaptureError(f"array {name} is not FP32")
        return values.astype(np.float64)

    def high_low_f64(self, high_name: str, low_name: str) -> np.ndarray:
        """Promote each high/low FP32 word, then sum in FP64."""
        high = self.f64(high_name)
        low = self.f64(low_name)
        if high.shape != low.shape:
            raise CaptureError("high/low arrays have different shapes")
        return high + low

    def fem_nodes_f64(self, name: str) -> dict[str, np.ndarray]:
        """Decode the frozen 80-byte NMFEMNodeStateGPU layout as exact FP32 words."""
        entry = self.arrays.get(name)
        if entry is None or entry.get("dtype") != "record:NMFEMNodeStateGPU":
            raise CaptureError(f"{name} is not an NMFEMNodeStateGPU array")
        if entry["element_bytes"] != 80:
            raise CaptureError("unexpected NMFEMNodeStateGPU byte layout")
        words = np.frombuffer(self.raw(name), dtype=np.dtype("<f4"))
        if words.size != entry["count"] * 20:
            raise CaptureError("NMFEMNodeStateGPU word count mismatch")
        fields = words.reshape(entry["count"], 5, 4).astype(np.float64)
        return {
            "position_and_mass": fields[:, 0, :],
            "velocity_and_inverse_mass": fields[:, 1, :],
            "rest_and_fixed": fields[:, 2, :],
            "delta_velocity": fields[:, 3, :],
            "reference_displacement_and_mode": fields[:, 4, :],
        }

    def summary(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "schema": m["schema"],
            "capture_point": m["capture_point"],
            "capture_kind": m["capture_kind"],
            "source_physics_fingerprint": m.get("source_physics_fingerprint"),
            "device_program_fingerprint": m.get("device_program_fingerprint"),
            "control_step": m.get("control_step"),
            "iteration": m.get("iteration"),
            "environment_count": m.get("environment_count"),
            "node_count": m.get("node_count"),
            "tetrahedron_count": m.get("tetrahedron_count"),
            "dt_seconds": m.get("dt_seconds"),
            "array_count": len(self.arrays),
            "payload_bytes": len(self.payload),
            "payload_sha256": sha256_bytes(self.payload),
            "array_names": sorted(self.arrays),
            "native_capture": True,
            "accepted_checkpoint": False,
            "physical_convergence": "not_assessed_by_capture_reader",
        }


def _matches(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pat) for pat in patterns)


def _selected_files(root: Path, patterns: list[str]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".coupled.bin") or relative.endswith(".coupled.json"):
            continue
        if _matches(relative, patterns):
            selected[relative] = path
    return selected


def compare_history_dirs(on_dir: Path, off_dir: Path,
                         patterns: list[str]) -> dict[str, Any]:
    on_files = _selected_files(on_dir, patterns)
    off_files = _selected_files(off_dir, patterns)
    missing_on = sorted(set(off_files) - set(on_files))
    missing_off = sorted(set(on_files) - set(off_files))
    files = []
    for relative in sorted(set(on_files) & set(off_files)):
        on_hash = sha256_bytes(on_files[relative].read_bytes())
        off_hash = sha256_bytes(off_files[relative].read_bytes())
        files.append({"path": relative, "on_sha256": on_hash,
                      "off_sha256": off_hash, "identical": on_hash == off_hash})
    identical = (bool(files) and not missing_on and not missing_off and
                 all(row["identical"] for row in files))
    return {
        "schema": "numi.matter.coupled_capture_history_comparison.v1",
        "diagnostic_on_directory": str(on_dir.resolve()),
        "diagnostic_off_directory": str(off_dir.resolve()),
        "selected_patterns": patterns,
        "files_compared": files,
        "missing_from_on": missing_on,
        "missing_from_off": missing_off,
        "raw_files_identical": identical,
        "physical_convergence": "not_assessed_by_byte_comparison",
        "claim_limit": "identical recorded histories do not admit a failed, incomplete, or unexecuted physical run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="validate a capture and print its summary")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--report", type=Path)
    compare = sub.add_parser("compare-histories",
        help="hash-compare selected raw production outputs from diagnostic on/off runs")
    compare.add_argument("--on-dir", type=Path, required=True)
    compare.add_argument("--off-dir", type=Path, required=True)
    compare.add_argument("--include", action="append", required=True,
                         help="relative glob; pass once per production history/checkpoint pattern")
    compare.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = Capture(args.manifest).summary()
            output = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if args.report:
                args.report.write_text(output)
            print(output, end="")
            return 0
        if not args.on_dir.is_dir() or not args.off_dir.is_dir():
            raise CaptureError("both on/off output directories must exist")
        result = compare_history_dirs(args.on_dir, args.off_dir, args.include)
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        args.report.write_text(output)
        print(output, end="")
        return 0 if result["raw_files_identical"] else 2
    except (CaptureError, OSError, json.JSONDecodeError) as exc:
        print(f"capture_reader_error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
