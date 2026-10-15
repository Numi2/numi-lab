#!/usr/bin/env python3
"""CPU-only integrity check for the source101 operator replay package."""
import hashlib
import json
import pathlib
import struct
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
PREP = ROOT / "prepared"
BASE = pathlib.Path("/Users/home/cardboard-implementation-20261014")
RUN = BASE / "solver/reset-augmentation-first-failure-capture-360"
SOURCE = BASE / "snapshots/source-boundary-reset-order-101"
BUILD = pathlib.Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-101")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise SystemExit("FAIL: " + message)


manifest = json.loads((PREP / "input-manifest.json").read_text())
seal = json.loads((PREP / "input-manifest.seal.json").read_text())
manifest_bytes = (PREP / "input-manifest.json").read_bytes()
payload = PREP / "replay-input.bin"
require(sha(PREP / "input-manifest.json") == seal["input_manifest_sha256"],
        "input manifest seal mismatch")
require(sha(payload) == manifest["payload_sha256"], "payload SHA mismatch")
require(payload.stat().st_size == manifest["payload_bytes"] == 2_054_152,
        "payload byte count mismatch")
raw = payload.read_bytes()
header = struct.unpack_from("<8s6IQfI", raw)
magic, version, envs, nodes, tets, state_stride, parameter_count, fingerprint, dt, _ = header
require((magic, version, envs, nodes, tets, state_stride, parameter_count) ==
        (b"NMCACHE1", 1, 2, 882, 1920, 15, 59), "payload header mismatch")
require(fingerprint == int("6de3475316b7e8b9", 16), "world fingerprint mismatch")
require(dt == manifest["native_capture"]["dt_seconds"], "dt mismatch")
require(sha(RUN / "operator-capture") == manifest["capture_sha256"], "capture changed")
require(sha(RUN / "output/compiled.nmatterpack") == manifest["compiled_package_sha256"],
        "compiled package changed")
require(sha(RUN / "inputs/runtime/matter/shaders/NumiMatter.metallib") ==
        manifest["metallib_sha256"], "metallib changed")
for relative, digest in manifest["source_hashes"].items():
    require(sha(SOURCE / relative) == digest, "source changed: " + relative)
binary = ROOT / "build/cached_operator_replay"
require(binary.is_file() and sha(binary) ==
        "d09b460ff73f2736da6c8209ca8ff5fcf0750ed2cbf57844148dd080110a7514",
        "compiled executable hash mismatch")
probe = subprocess.run([
    str(binary), str(RUN / "output/compiled.nmatterpack"), str(payload),
    "/tmp/cardboard-operator-preflight-only", "--preflight-only",
], capture_output=True, text=True, check=False)
require(probe.returncode == 0, "compiled immutable-field audit failed: " + probe.stderr.strip())
require("no Metal device or pipeline was created" in probe.stdout,
        "compiled audit did not use CPU-only preflight mode")
print("PASS source101 replay CPU preflight: sealed hashes and 3840 immutable tetrahedron records match; no Metal device created")
