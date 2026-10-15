#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).with_name("read_selector5_capture.py")
spec = importlib.util.spec_from_file_location("selector5_reader", SCRIPT)
reader = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(reader)


def f32s(values):
    return struct.pack("<" + "f" * len(values), *values)


def u32s(values):
    return struct.pack("<" + "I" * len(values), *values)


def synthetic_capture():
    # One environment, one free node, one tet, one coarse mode, one graph patch.
    node_state = [0.0] * 20
    node_state[3] = 1.0  # mass
    node_state[11] = 0.0  # restAndFixed.w: free
    rows = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0]
    patch_words = [0, 1, 0, 0]
    arrays = [
        ("P", "<f4x4", [1, 1], f32s([1, 0, 0, 0])),
        ("effective_P", "<f4x4", [1, 1, 1], f32s([1, 0, 0, 0])),
        ("AP", "<f4x4", [1, 1, 1], f32s([1, 0, 0, 0])),
        ("pivoted_RAP", "<f4", [1, 1, 1], f32s([1])),
        ("pivots", "<u4", [1, 1], u32s([0])),
        ("factor_valid", "<u4", [1], u32s([1])),
        ("candidate_nodes", "struct:NMFEMNodeStateGPU", [1, 1], f32s(node_state)),
        ("material_history", "<f4", [1, 1, 1], f32s([0])),
        ("scheduler_exponents", "<u4", [1, 1], u32s([0])),
        ("deformation_high", "<f4x4", [1, 1, 3], f32s([1, 0, 0, 0] * 3)),
        ("deformation_low", "<f4x4", [1, 1, 3], f32s([0, 0, 0, 0] * 3)),
        ("production_status", "struct:NMMatterStatusGPU", [1], bytes(48)),
        ("graph_patches", "<u4x4", [1], u32s(patch_words)),
        ("graph_patch_nodes", "<u4", [1], u32s([0])),
        ("graph_rows", "<u4x4", [3], u32s(rows)),
        ("graph_node_patch_offsets", "<u4", [2], u32s([0, 1])),
        ("graph_node_patch_values", "<u4", [1], u32s([0])),
        ("graph_node_coverage", "<u4", [1], u32s([1])),
        ("graph_patch_matrices_lu", "<f4", [9], f32s([1, 0, 0, 0, 1, 0, 0, 0, 1])),
        ("graph_pivots", "<u4", [3], u32s([0, 1, 2])),
        ("graph_factor_valid", "<u4", [1], u32s([1])),
    ]
    payload = bytearray()
    descriptors = []
    blobs = {}
    for name, dtype, shape, data in arrays:
        blobs[name] = data
        descriptors.append({"name": name, "dtype": dtype, "shape": shape,
                            "offset": len(payload), "byte_count": len(data),
                            "element_bytes": reader.DTYPE_BYTES[dtype],
                            "sha256": hashlib.sha256(data).hexdigest()})
        payload.extend(data)
    digest = hashlib.sha256()
    for name in ("candidate_nodes", "material_history", "scheduler_exponents",
                 "deformation_high", "deformation_low"):
        digest.update(blobs[name])
    manifest = {
        "schema": reader.SCHEMA, "byte_order": "little_endian",
        "capture_point": reader.CAPTURE_POINT, "preconditioner_selector": 5,
        "root": 3, "iteration": 7, "environment_count": 1,
        "node_count": 1, "tetrahedron_count": 1, "mode_count": 1,
        "material_state_stride": 1, "micro_generation": 2,
        "linearization_generation": (2 << 32) | 7,
        "tangent_cache_used": False,
        "source_physics_fingerprint": "0000000000000001",
        "device_program_fingerprint": "0000000000000002",
        "graph_schwarz": {"enabled": True, "capture_environment": 0,
            "patch_count": 1, "patch_node_count": 1, "dof_count": 3,
            "matrix_float_count": 9, "pivot_count": 3,
            "layout_checksum_fnv1a32": 0,
            "source_topology_sha256": "a" * 64,
            "source_free_mask_sha256": hashlib.sha256(bytes([1])).hexdigest(),
            "source_topology_digest_encoding": "little-endian u32 node_count, u32 tetrahedron_count, then row-major u32 tet connectivity; free mask is raw u8, 1=free 0=fixed"},
        "candidate_linearization_sha256": [{"environment": 0,
            "sha256": digest.hexdigest(),
            "candidate_nodes_sha256": hashlib.sha256(blobs["candidate_nodes"]).hexdigest()}],
        "payload": {"byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()},
        "arrays": descriptors,
    }
    checksum = 2166136261
    for value in patch_words + rows + [0] + [0, 1] + [0]:
        checksum = ((checksum ^ value) * 16777619) & 0xffffffff
    manifest["graph_schwarz"]["layout_checksum_fnv1a32"] = checksum
    return manifest, bytes(payload)


class Selector5ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest, self.payload = synthetic_capture()
        self.manifest_path = self.root / "capture.json"
        self.payload_path = self.root / "capture.bin"
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.manifest["payload"] = {"byte_count": len(self.payload),
                                    "sha256": hashlib.sha256(self.payload).hexdigest()}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.payload_path.write_bytes(self.payload)

    def test_valid_minimal_capture(self):
        result = reader.validate_capture(self.manifest_path, self.payload_path)
        self.assertTrue(result["all_graph_factors_valid"])
        self.assertEqual(result["topology_validation"]["status"], "source_hash_present_topology_input_missing")
        self.assertEqual(result["patch_count"], 1)
        self.assertEqual(result["iteration"], 7)
        _, arrays, _ = reader.load_validated_arrays(self.manifest_path, self.payload_path)
        self.assertEqual(arrays["graph_patches"].shape, (1, 4))
        self.assertEqual(arrays["graph_patch_matrices_lu"].shape, (9,))
        self.assertEqual(arrays["candidate_nodes"].shape, (1, 1, 80))

    def test_require_topology_fails_closed_without_bound_artifact(self):
        with self.assertRaisesRegex(ValueError, "exact topology input is required"):
            reader.validate_capture(self.manifest_path, self.payload_path, require_topology=True)
        with tempfile.TemporaryDirectory() as directory:
            topology = Path(directory) / "topology.npz"
            np.savez(topology, node_count=np.array(1, dtype="<u4"),
                     nodes=np.empty((0, 4), dtype="<u4"))
            self.manifest["graph_schwarz"].pop("source_topology_sha256")
            self.write()
            with self.assertRaisesRegex(ValueError, "source topology/free-mask hashes"):
                reader.validate_capture(self.manifest_path, self.payload_path,
                                        topology_npz=topology)

    def test_exact_topology_map_rejects_same_count_wrong_global_ids(self):
        tetrahedra = np.array([[0, 1, 2, 3]], dtype="<u4")
        topology = reader.rebuild_graph_layout(tetrahedra, [True, True, True, True])
        self.assertEqual(topology["patches"], [[0, 4, 0, 0], [4, 4, 144, 12]])
        self.assertEqual(topology["patch_nodes"], [0, 1, 2, 3, 0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as directory:
            topology_path = Path(directory) / "topology.npz"
            np.savez(topology_path, node_count=np.array(4, dtype="<u4"), nodes=tetrahedra)
            raw = topology_path.read_bytes()
            free_digest = hashlib.sha256(bytes([1, 1, 1, 1])).hexdigest()
            topology_digest = hashlib.sha256(struct.pack("<II", 4, 1) + tetrahedra.tobytes()).hexdigest()
            manifest = {
                "node_count": 4, "tetrahedron_count": 1,
                "source_physics_fingerprint": "physics", "device_program_fingerprint": "device",
                "graph_schwarz": {
                    "source_topology_sha256": topology_digest,
                    "source_free_mask_sha256": free_digest,
                    "source_topology_digest_encoding": "little-endian u32 node_count, u32 tetrahedron_count, then row-major u32 tet connectivity; free mask is raw u8, 1=free 0=fixed",
                },
            }
            mapping = {
                "patches": ("<u4x4", [2, 4]), "patch_nodes": ("<u4", [8]),
                "rows": ("<u4x4", [24, 4]), "node_patch_offsets": ("<u4", [5]),
                "node_patch_values": ("<u4", [8]), "node_coverage": ("<u4", [4]),
            }
            arrays = {}
            for name, (dtype, shape) in mapping.items():
                data_name = {"patches": "patches", "patch_nodes": "patch_nodes", "rows": "rows",
                             "node_patch_offsets": "node_patch_offsets", "node_patch_values": "node_patch_values",
                             "node_coverage": "node_coverage"}[name]
                values = topology[data_name]
                blob = np.asarray(values, dtype="<u4").tobytes()
                arrays["graph_" + name] = {"descriptor": {"dtype": dtype, "shape": shape}, "bytes": blob}
            checksum = reader.fnv1a_layout(topology)
            fixed_tags = [0., 0., 0., 0.]
            result = reader._verify_bound_topology(manifest, topology_path,
                fixed_tags, arrays, checksum)
            self.assertEqual(result["status"], "topology_and_global_ids_verified")

            alternate_path = Path(directory) / "same_counts_other_tet_order.npz"
            alternate = np.array([[0, 2, 1, 3]], dtype="<u4")
            np.savez(alternate_path, node_count=np.array(4, dtype="<u4"), nodes=alternate)
            with self.assertRaisesRegex(ValueError, "NPZ topology differs from producer"):
                reader._verify_bound_topology(manifest, alternate_path,
                    fixed_tags, arrays, checksum)

            wrong = dict(arrays)
            wrong["graph_patch_nodes"] = {
                "descriptor": arrays["graph_patch_nodes"]["descriptor"],
                "bytes": np.array([1, 0, 2, 3, 0, 1, 2, 3], dtype="<u4").tobytes(),
            }
            with self.assertRaisesRegex(ValueError, "exact topology-derived global-ID map"):
                reader._verify_bound_topology(manifest, topology_path,
                    fixed_tags, wrong, checksum)

    def test_bound_topology_requires_native_binary_fixed_tags_and_integer_count(self):
        tetrahedra = np.array([[0, 1, 2, 3]], dtype="<u4")
        topology = reader.rebuild_graph_layout(tetrahedra, [True, True, True, True])
        with tempfile.TemporaryDirectory() as directory:
            topology_path = Path(directory) / "topology.npz"
            np.savez(topology_path, node_count=np.array(4, dtype="<u4"), nodes=tetrahedra)
            free_digest = hashlib.sha256(bytes([1, 1, 1, 1])).hexdigest()
            topology_digest = hashlib.sha256(struct.pack("<II", 4, 1) + tetrahedra.tobytes()).hexdigest()
            manifest = {"node_count": 4, "tetrahedron_count": 1,
                "graph_schwarz": {"source_topology_sha256": topology_digest,
                    "source_free_mask_sha256": free_digest,
                    "source_topology_digest_encoding": "little-endian u32 node_count, u32 tetrahedron_count, then row-major u32 tet connectivity; free mask is raw u8, 1=free 0=fixed"}}
            arrays = {}
            for key, name in (("patches", "graph_patches"), ("patch_nodes", "graph_patch_nodes"),
                              ("rows", "graph_rows"), ("node_patch_offsets", "graph_node_patch_offsets"),
                              ("node_patch_values", "graph_node_patch_values"), ("node_coverage", "graph_node_coverage")):
                values = topology[key]
                blob = np.asarray(values, dtype="<u4").tobytes()
                arrays[name] = {"descriptor": {"dtype": "<u4x4" if key in ("patches", "rows") else "<u4"}, "bytes": blob}
            checksum = reader.fnv1a_layout(topology)
            with self.assertRaisesRegex(ValueError, "exact binary source values"):
                reader._verify_bound_topology(manifest, topology_path,
                    [0.25, 0.0, 0.0, 0.0], arrays, checksum)
            np.savez(topology_path, node_count=np.array(4.5, dtype="<f8"), nodes=tetrahedra)
            with self.assertRaisesRegex(ValueError, "scalar integer"):
                reader._verify_bound_topology(manifest, topology_path,
                    [0.0, 0.0, 0.0, 0.0], arrays, checksum)

    def test_reject_wrong_selector_or_capture_point(self):
        for key, value in (("preconditioner_selector", 3),
                           ("capture_point", "after_factor")):
            original = self.manifest[key]
            self.manifest[key] = value
            self.write()
            with self.assertRaises(ValueError):
                reader.validate_capture(self.manifest_path, self.payload_path)
            self.manifest[key] = original

    def test_reject_payload_and_array_hash_tampering(self):
        self.payload_path.write_bytes(self.payload[:-1] + bytes([self.payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "payload length or SHA-256"):
            reader.validate_capture(self.manifest_path, self.payload_path)
        self.write()
        self.manifest["arrays"][0]["sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            reader.validate_capture(self.manifest_path, self.payload_path)

    def test_reject_generation_digest_and_environment_mismatch(self):
        self.manifest["linearization_generation"] += 1
        self.write()
        with self.assertRaisesRegex(ValueError, "generation encoding"):
            reader.validate_capture(self.manifest_path, self.payload_path)
        self.manifest["linearization_generation"] -= 1
        self.manifest["candidate_linearization_sha256"][0]["sha256"] = "f" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "candidate/linearization digest"):
            reader.validate_capture(self.manifest_path, self.payload_path)
        self.manifest["candidate_linearization_sha256"][0]["sha256"] = synthetic_capture()[0]["candidate_linearization_sha256"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "selected environment"):
            reader.validate_capture(self.manifest_path, self.payload_path, 1)

    def test_reject_graph_coverage_factor_and_checksum(self):
        bad_payload = bytearray(self.payload)
        desc = next(x for x in self.manifest["arrays"] if x["name"] == "graph_node_coverage")
        bad_payload[desc["offset"]:desc["offset"] + 4] = u32s([0])
        self.payload = bytes(bad_payload)
        self.write()
        with self.assertRaisesRegex(ValueError, "coverage"):
            reader.validate_capture(self.manifest_path, self.payload_path)
        self.payload = synthetic_capture()[1]
        self.manifest["graph_schwarz"]["layout_checksum_fnv1a32"] ^= 1
        self.write()
        with self.assertRaisesRegex(ValueError, "FNV-1a checksum"):
            reader.validate_capture(self.manifest_path, self.payload_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
