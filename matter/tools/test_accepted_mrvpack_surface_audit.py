import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path


SCRIPT = Path(__file__).with_name("accepted_mrvpack_surface_audit.py")
SPEC = importlib.util.spec_from_file_location("accepted_surface_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def write_pack(path, vertices_xyz):
    header_size = audit.HEADER.size
    directory_size = audit.DIRECTORY.size
    section_start = header_size + directory_size * 3
    vertices = bytearray(len(vertices_xyz) * 80)
    for index, xyz in enumerate(vertices_xyz):
        struct.pack_into("<3f", vertices, index * 80, *xyz)
    indices = struct.pack("<3I", 0, 1, 2)
    primitive = bytearray(64)
    struct.pack_into("<4I", primitive, 0, 0, 3, 0, 0)
    struct.pack_into("<4I", primitive, 16, 51023, 0, 0, 305)
    sections = ((2, bytes(vertices), len(vertices_xyz), 80),
                (3, indices, 3, 4),
                (4, bytes(primitive), 1, 64))
    offsets = []
    cursor = section_start
    for _, data, _, _ in sections:
        offsets.append(cursor)
        cursor += len(data)
    header = audit.HEADER.pack(b"MRVPACK2", 2, 3, 0, 0, b"\0" * 32, b"\0" * 24)
    directories = b"".join(
        audit.DIRECTORY.pack(kind, 0, offset, len(data), count, stride, 0, b"\0" * 32)
        for (kind, data, count, stride), offset in zip(sections, offsets))
    path.write_bytes(header + directories + b"".join(row[1] for row in sections))
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "accepted_pack_path": str(path.resolve()),
        "accepted_step": 639,
        "accepted_time_s": 1.278,
        "accepted_root_fingerprint": 123456789,
        "accepted_transaction_fingerprint": 123456789,
        "accepted_body_state_sha256": "body-state-digest",
        "accepted_respiration_state_sha256": "resp-state-digest",
        "vertex_count": len(vertices_xyz),
        "index_count": 3,
        "captured_vertex_buffer_sha256": hashlib.sha256(vertices).hexdigest(),
        "accepted_registered_body_poses": [{
            "body_index": 20,
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }],
    }
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path


class AcceptedMrvpackAuditTests(unittest.TestCase):
    def test_receipt_binds_exact_pack_and_vertex_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            pack = root / "step-639.mrvpack"
            receipt = write_pack(pack, points)
            mapped, stream, vertex_offset, surfaces = audit.read_pack(pack)
            try:
                provenance = audit.validate_accepted_receipt(
                    pack, receipt, 639, mapped, vertex_offset, surfaces)
                with self.assertRaisesRegex(ValueError, "step"):
                    audit.validate_accepted_receipt(
                        pack, receipt, 640, mapped, vertex_offset, surfaces)
            finally:
                mapped.close()
                stream.close()
            self.assertEqual(provenance["accepted_step"], 639)
            self.assertEqual(provenance["accepted_root_fingerprint"], 123456789)
            self.assertEqual(provenance["captured_vertex_buffer_sha256"],
                             hashlib.sha256(pack.read_bytes()[
                                 audit.HEADER.size + 3 * audit.DIRECTORY.size:
                                 audit.HEADER.size + 3 * audit.DIRECTORY.size + 240]).hexdigest())
            self.assertEqual(provenance["sha256"], hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_swapped_receipt_pack_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mrvpack"
            second = root / "second.mrvpack"
            receipt = write_pack(first, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            write_pack(second, [(0, 0, 0), (2, 0, 0), (0, 1, 0)])
            mapped, stream, vertex_offset, surfaces = audit.read_pack(second)
            try:
                with self.assertRaisesRegex(ValueError, "pack path"):
                    audit.validate_accepted_receipt(second, receipt, 639,
                                                    mapped, vertex_offset, surfaces)
            finally:
                mapped.close()
                stream.close()

    def test_repointed_receipt_with_wrong_vertex_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mrvpack"
            second = root / "second.mrvpack"
            receipt = write_pack(first, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            write_pack(second, [(0, 0, 0), (2, 0, 0), (0, 1, 0)])
            data = json.loads(receipt.read_text())
            data["accepted_pack_path"] = str(second.resolve())
            repointed = root / "repointed.receipt.json"
            repointed.write_text(json.dumps(data))
            mapped, stream, vertex_offset, surfaces = audit.read_pack(second)
            try:
                with self.assertRaisesRegex(ValueError, "vertex-buffer SHA-256"):
                    audit.validate_accepted_receipt(second, repointed, 639,
                                                    mapped, vertex_offset, surfaces)
            finally:
                mapped.close()
                stream.close()

    def test_wrong_predicate_module_file_is_not_reused_from_import_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "predicate_tree"
            package.mkdir()
            (package / "__init__.py").write_text("\"\"\"test package\"\"\"\n")
            source = package / "predicate.py"
            source.write_text(
                "marker = 'exact-source'\n"
                "def _records(*args): return []\n"
                "def _audit_pair(*args, **kwargs): return {}\n"
                "def triangle_intersection_points(*args): return []\n")
            wrong = types.ModuleType("numilab_human.cardiac_cavity_intersections")
            wrong.__file__ = str(root / "wrong.py")
            previous = sys.modules.get(wrong.__name__)
            sys.modules[wrong.__name__] = wrong
            try:
                loaded = audit.predicate_module(source)
            finally:
                if previous is None:
                    sys.modules.pop(wrong.__name__, None)
                else:
                    sys.modules[wrong.__name__] = previous
            self.assertEqual(Path(loaded.__file__).resolve(), source.resolve())
            self.assertEqual(loaded.marker, "exact-source")

    def test_coordinate_lattice_preserves_all_binary32_values(self):
        values = [f32(0.1), f32(-0.2), f32(2.0 ** -100), f32(-0.0)]
        scale = audit.coordinate_lattice_scale(values)
        self.assertGreater(scale, 2 ** 90)
        for value in values:
            encoded = audit.lattice_integer(value, scale)
            self.assertEqual(Fraction(encoded, scale), Fraction.from_float(value))
        with self.assertRaisesRegex(ValueError, "common lattice"):
            audit.lattice_integer(values[0], scale // (2 ** 80))

    def test_zero_area_triangle_is_reported_and_never_filtered(self):
        repeated = f32(0.03303375840187073)
        points = {
            847665: (repeated, f32(-0.48139631748199463), f32(0.17360453307628632)),
            847667: (repeated, f32(-0.48139631748199463), f32(0.17360453307628632)),
            847670: (f32(0.033033765852451324), f32(-0.48139631748199463),
                    f32(0.1736045777797699)),
        }
        scale = audit.coordinate_lattice_scale(value for xyz in points.values() for value in xyz)

        class ExactPredicates:
            @staticmethod
            def _records(vertices, faces):
                records = []
                for face_index, ids in enumerate(faces):
                    tri = tuple(vertices[index] for index in ids)
                    ab = tuple(tri[1][axis] - tri[0][axis] for axis in range(3))
                    ac = tuple(tri[2][axis] - tri[0][axis] for axis in range(3))
                    cross = (ab[1] * ac[2] - ab[2] * ac[1],
                             ab[2] * ac[0] - ab[0] * ac[2],
                             ab[0] * ac[1] - ab[1] * ac[0])
                    if not any(cross):
                        raise ValueError("exactly degenerate triangle")
                    records.append((tri, (), (), face_index, tuple(ids)))
                return records

        with self.assertRaisesRegex(ValueError, r"zero-area triangle rows \[0\].*no faces were omitted"):
            audit.make_records((list(points), points, scale), scale,
                               [(847665, 847667, 847670)], ExactPredicates)

    def test_cli_emits_receipt_bound_v2_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "step-639.mrvpack"
            receipt = write_pack(pack, [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            package = root / "predicate_tree"
            package.mkdir()
            (package / "__init__.py").write_text("\"\"\"test package\"\"\"\n")
            predicate = package / "predicate.py"
            predicate.write_text(
                "def _records(vertices, faces): return []\n"
                "def _audit_pair(first, second, **kwargs):\n"
                " return {'triangle_pairs': [], 'count': 0, 'aabb_candidate_pairs': 0}\n"
                "def triangle_intersection_points(first, second): return []\n")
            original_argv = sys.argv
            sys.argv = [str(SCRIPT), str(pack), str(predicate), "--step", "639",
                        "--receipt", str(receipt), "--pairs", "51023:305-51023:305"]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    audit.main()
            finally:
                sys.argv = original_argv
            report = json.loads(output.getvalue())
            self.assertEqual(report["schema"], "accepted-mrvpack-pair-audit.v2")
            self.assertEqual(report["accepted_receipt"]["accepted_step"], 639)
            self.assertEqual(report["accepted_receipt"]["captured_vertex_buffer_sha256"],
                             json.loads(receipt.read_text())["captured_vertex_buffer_sha256"])
            self.assertIn("post-motion", report["limitation"])



class SharedSourceBoundaryTests(unittest.TestCase):
    left, right = (51010, 23), (51025, 319)

    def fixture(self):
        points = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
        source = {self.left: (points, [(0, 1, 2)]),
                  self.right: (points, [(0, 2, 1)])}
        surfaces = {self.left: {"faces": [(3, 5, 1)]},
                    self.right: {"faces": [(0, 4, 2)]}}
        capture = bytearray(6 * 80)
        # Different record order and opposite winding retain explicit incidence.
        for index, point in zip((3, 5, 1, 0, 2, 4), points + points):
            struct.pack_into("<3f", capture, index * 80, *point)
        return capture, surfaces, source

    def test_shared_points_survive_noncontiguous_native_vertex_indices(self):
        capture, surfaces, source = self.fixture()
        row = audit.shared_source_boundaries(capture, 0, surfaces, source,
                                             [(self.left, self.right)])[0]
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["source_shared_coordinate_count"], 3)
        self.assertEqual(row["native_different_vertex_copies"], 0)

    def test_one_ulp_gap_is_reported_without_tolerance(self):
        capture, surfaces, source = self.fixture()
        value = struct.unpack("<f", struct.pack("<I", 0x3f800001))[0]
        struct.pack_into("<f", capture, 2 * 80, value)
        row = audit.shared_source_boundaries(capture, 0, surfaces, source,
                                             [(self.left, self.right)])[0]
        self.assertEqual(row["status"], "different_native_boundary_copies")
        self.assertEqual(row["native_different_vertex_copies"], 1)
        self.assertEqual(row["max_world_gap_m"], 2. ** -23)
        self.assertEqual(row["witnesses"][0]["second_local_vertex"], 1)

    def test_topology_disagreement_fails_closed(self):
        capture, surfaces, source = self.fixture()
        source[self.left] = (source[self.left][0], [(0, 1, 2), (0, 2, 1)])
        surfaces[self.left]["faces"] = [(3, 5, 1), (5, 1, 3)]
        with self.assertRaisesRegex(ValueError, "not bijective"):
            audit.shared_source_boundaries(capture, 0, surfaces, source,
                                            [(self.left, self.right)])

    def test_missing_or_wrong_count_source_fails_closed(self):
        capture, surfaces, source = self.fixture()
        missing = dict(source); del missing[self.right]
        with self.assertRaisesRegex(ValueError, "identity absent"):
            audit.shared_source_boundaries(capture, 0, surfaces, missing,
                                            [(self.left, self.right)])
        source[self.left] = (source[self.left][0], [])
        with self.assertRaisesRegex(ValueError, "triangle counts"):
            audit.shared_source_boundaries(capture, 0, surfaces, source,
                                            [(self.left, self.right)])

    def test_no_common_points_does_not_pass_vacuously(self):
        capture, surfaces, source = self.fixture()
        source[self.right] = ([(2., 0., 0.), (3., 0., 0.), (2., 1., 0.)],
                              source[self.right][1])
        row = audit.shared_source_boundaries(capture, 0, surfaces, source,
                                             [(self.left, self.right)])[0]
        self.assertEqual(row["status"], "no_shared_source_boundary")
        self.assertIsNone(row["rms_world_gap_m"])

    def test_nonfinite_native_vertex_fails_closed(self):
        capture, surfaces, source = self.fixture()
        struct.pack_into("<f", capture, 0, float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit.shared_source_boundaries(capture, 0, surfaces, source,
                                            [(self.left, self.right)])

    def test_duplicate_source_point_copies_are_all_checked(self):
        capture, surfaces, source = self.fixture()
        points, faces = source[self.left]
        source[self.left] = (points + [points[0]], faces + [(3, 1, 2)])
        surfaces[self.left]["faces"].append((6, 5, 1))
        capture.extend(bytearray(80))
        struct.pack_into("<3f", capture, 6 * 80, 0., 0., 2. ** -100)
        row = audit.shared_source_boundaries(capture, 0, surfaces, source,
                                             [(self.left, self.right)])[0]
        self.assertEqual(row["native_vertex_copy_comparisons"], 4)
        self.assertEqual(row["native_different_vertex_copies"], 1)

    def test_source_identity_mismatch_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.nhanatomy"
            path.write_bytes(b"not the declared source")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit.load_source_boundary_meshes(path, "0" * 64, {}, set())


if __name__ == "__main__":
    unittest.main()
