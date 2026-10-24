#!/usr/bin/env python3
import hashlib
import importlib.util
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "payload" / "external-inputs" / "rebind_human_tissue_reference.py"
TOOL = ROOT / "tools" / "replace_surface_topology.py"
spec = importlib.util.spec_from_file_location("replace_surface_topology_052", TOOL)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
canon_spec = importlib.util.spec_from_file_location("canonical_nhtiss_owner_396_test", CANON)
canon = importlib.util.module_from_spec(canon_spec)
sys.modules[canon_spec.name] = canon
canon_spec.loader.exec_module(canon)


def fixture():
    nr, nb, nv = 3, 3, 15
    # A target square disk with diagonal 0-2. It is replaced by a four-triangle
    # fan about an already existing fifth vertex; stable 136 follows the target
    # to check index offsets are rebuilt.
    faces = [
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        np.asarray([[5, 6, 7], [5, 7, 8]], dtype=np.uint32),
        np.asarray([[10, 11, 12]], dtype=np.uint32),
    ]
    records = np.asarray([
        [0, 1, 0, 5, 0, 3, 132, 1],
        [1, 1, 5, 5, 3, 6, 134, 1],
        [2, 1, 10, 5, 9, 3, 136, 1],
    ], dtype="<u4")
    bindings = np.zeros(nb, dtype=canon.BIND_DTYPE)
    vertices = np.zeros(nv, dtype=canon.VERT_DTYPE)
    for i in range(nb):
        bindings[i]["body"] = 900 + i
        bindings[i]["transform"] = np.asarray([0, 0, 0, 0, 0, 0, 1, .001], dtype=np.float32)
    for i in range(nv):
        vertices[i]["p"] = np.asarray([i * .01, i * .02, i * .03], dtype=np.float32)
        vertices[i]["n"] = np.asarray([0, 0, 1], dtype=np.float32)
        vertices[i]["binding"] = np.asarray([0, 0, 0, 0], dtype=np.uint32)
        vertices[i]["w"] = np.asarray([1, 0, 0, 0], dtype=np.float32)
    indices = np.concatenate([f.reshape(-1) for f in faces]).astype("<u4")
    raw = canon.HEADER.pack(b"NHTISS4\0", 5, nr, nb, nv, len(indices), 77, bytes(range(32)))
    raw += records.tobytes() + bindings.tobytes() + vertices.tobytes() + indices.tobytes()
    return raw


class TopologyReplacementTests(unittest.TestCase):
    def args(self):
        return dict(
            stable_id=134,
            patch_face_rows=np.asarray([0, 1], dtype=np.uint32),
            fan_faces_local=np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.uint32),
            boundary_cycle=np.asarray([0, 1, 2, 3], dtype=np.uint32),
            ancestry_offsets=np.asarray([0, 2, 4, 6, 8], dtype=np.uint32),
            ancestry_face_rows=np.asarray([0, 1] * 4, dtype=np.uint32),
        )

    def test_face_count_offsets_and_non_topology_payload_are_rebuilt(self):
        raw = fixture()
        old = canon.decode_nhtiss(raw)
        out_raw, report = module.replace_surface_topology(raw, canon, **self.args())
        out = canon.decode_nhtiss(out_raw)
        self.assertEqual(report["status"], "candidate_only_mechanical_face_map_invalid")
        self.assertEqual(out["ni"], old["ni"] + 6)
        self.assertEqual(int(out["records"][1, 5]), 12)
        self.assertEqual(int(out["records"][2, 4]), 15)
        self.assertEqual(int(out["records"][2, 5]), 3)
        self.assertTrue(np.array_equal(out["bindings"], old["bindings"]))
        self.assertTrue(np.array_equal(out["vertices"], old["vertices"]))
        self.assertEqual(out["registration"], old["registration"])
        self.assertEqual(out["source_id"], old["source_id"])
        for ri in (0, 2):
            oldrow, newrow = old["records"][ri], out["records"][ri]
            a = old["indices"][int(oldrow[4]):int(oldrow[4] + oldrow[5])]
            b = out["indices"][int(newrow[4]):int(newrow[4] + newrow[5])]
            self.assertTrue(np.array_equal(a, b))
        target = out["records"][1]
        target_faces = out["indices"][int(target[4]):int(target[4] + target[5])].reshape(-1, 3) - int(target[2])
        self.assertEqual(target_faces.tolist(), self.args()["fan_faces_local"].tolist())

    def test_rejects_bad_boundary_and_incomplete_construction_ancestry(self):
        raw = fixture()
        bad = self.args()
        bad["boundary_cycle"] = np.asarray([0, 1, 3, 2], dtype=np.uint32)
        with self.assertRaises(module.TopologyError):
            module.replace_surface_topology(raw, canon, **bad)
        bad = self.args()
        bad["ancestry_face_rows"] = np.asarray([0] * 8, dtype=np.uint32)
        with self.assertRaises(module.TopologyError):
            module.replace_surface_topology(raw, canon, **bad)

    def test_cli_bad_input_pin_emits_no_candidate(self):
        import json
        import tempfile
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.nhtissue"
            source.write_bytes(fixture())
            side = root / "patch.npz"
            np.savez(side, stable_id=np.asarray([134], dtype=np.uint32),
                     source_patch_face_rows=self.args()["patch_face_rows"],
                     fan_source_vertex_triangles=self.args()["fan_faces_local"],
                     fixed_boundary_cycle=self.args()["boundary_cycle"],
                     construction_ancestry_offsets=self.args()["ancestry_offsets"],
                     construction_ancestry_source_face_rows=self.args()["ancestry_face_rows"])
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"payload": {}, "source": {}}, sort_keys=True))
            outputs = [root / "out.nhtissue", root / "derived.json", root / "report.json"]
            argv = [
                "--source", str(source), "--source-sha256", "0" * 64,
                "--sidecar", str(side), "--sidecar-sha256", hashlib.sha256(side.read_bytes()).hexdigest(),
                "--manifest", str(manifest), "--manifest-sha256", hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "--canonical-owner", str(CANON), "--canonical-owner-sha256", hashlib.sha256(CANON.read_bytes()).hexdigest(),
                "--stable-id", "134", "--output", str(outputs[0]), "--derived-manifest", str(outputs[1]),
                "--report", str(outputs[2]),
            ]
            with redirect_stderr(StringIO()):
                status = module.main(argv)
            self.assertEqual(status, 2)
            self.assertFalse(any(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()

