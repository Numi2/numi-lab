#!/usr/bin/env python3
"""Deck-generation checks only; these fixtures do not run or qualify Radioss."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from cardboard_opencourant_paper import (
    PARAMETERS,
    export_mechanics_fixtures,
    export_paper_cases,
)


FROZEN_PARAMETERS_SHA256 = "46d33c7e6e4c796ad3285e269df03b1a0a6442dc70e6cb86de8a38ab5568527b"
PAPER_CASE_IDS = [
    "law112-md-tension-instrument",
    "law112-md-compression-instrument",
    "law112-cd-tension-instrument",
    "law112-cd-compression-instrument",
]


def _law_card(starter: str) -> str:
    return starter.split("/MAT/LAW112/1\n", 1)[1].split("/PROP/TYPE6/1\n", 1)[0]


def _node_groups(starter: str) -> dict[int, list[int]]:
    lines = starter.splitlines()
    groups = {}
    for index, line in enumerate(lines):
        if line.startswith("/GRNOD/NODE/"):
            groups[int(line.rsplit("/", 1)[1])] = [int(node) for node in lines[index + 2].split()]
    return groups


def _bcs(starter: str) -> list[tuple[str, int]]:
    lines = starter.splitlines()
    cards = []
    for index, line in enumerate(lines):
        if line.startswith("/BCS/"):
            data = lines[index + 2]
            cards.append((data[3:6] + data[7:10], int(data[20:30])))
    return cards


def _function_points(starter: str, function_id: int) -> list[tuple[float, float]]:
    lines = starter.splitlines()
    marker = f"/FUNCT/{function_id}"
    index = lines.index(marker) + 2
    points = []
    while index < len(lines) and not lines[index].startswith("/"):
        if lines[index].strip():
            t, value = (float(field) for field in lines[index].split())
            points.append((t, value))
        index += 1
    return points


def _rank(rows: list[list[float]], tolerance: float = 1.0e-10) -> int:
    matrix = [row[:] for row in rows]
    rank = 0
    for column in range(len(matrix[0])):
        pivot = max(range(rank, len(matrix)), key=lambda row: abs(matrix[row][column]),
                    default=rank)
        if rank >= len(matrix) or abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(len(matrix)):
            if row != rank:
                factor = matrix[row][column]
                matrix[row] = [value - factor * lead
                               for value, lead in zip(matrix[row], matrix[rank])]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _rigid_mode_row(point: tuple[float, float, float], direction: tuple[float, float, float]) -> list[float]:
    x, y, z = point
    dx, dy, dz = direction
    return [dx, dy, dz, -dy * z + dz * y, dx * z - dz * x, -dx * y + dy * x]


class PaperMechanicsDecks(unittest.TestCase):
    def test_default_four_direction_decks_and_parameter_card_stay_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "paper"
            output.mkdir()
            cases = export_paper_cases(output)

            self.assertEqual([case["id"] for case in cases], PAPER_CASE_IDS)
            parameter_sha = hashlib.sha256(PARAMETERS.read_bytes()).hexdigest()
            self.assertEqual(parameter_sha, FROZEN_PARAMETERS_SHA256)
            for case in cases:
                self.assertFalse(case["physical_qualification"])
                self.assertEqual(case["calibration_scope"], "instrument_only_not_target_grade_calibration")
                self.assertEqual(case["parameters"]["sha256"], FROZEN_PARAMETERS_SHA256)
                starter = output / case["id"] / case["starter"]
                self.assertTrue(starter.is_file())

    def test_opt_in_mechanics_fixtures_use_same_law_and_native_kinematics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper_dir = root / "paper"
            mechanics_dir = root / "mechanics"
            paper_dir.mkdir()
            mechanics_dir.mkdir()
            paper_cases = export_paper_cases(paper_dir)
            cases = export_mechanics_fixtures(mechanics_dir)

            self.assertEqual([case["id"] for case in paper_cases], PAPER_CASE_IDS)
            self.assertEqual(
                [case["id"] for case in cases],
                ["law112-md-cd-shear-elastic-instrument",
                 "law112-md-tension-plastic-reversal-instrument",
                 "law112-rigid-rotation-objectivity-instrument"],
            )
            baseline = (paper_dir / PAPER_CASE_IDS[0] / paper_cases[0]["starter"]).read_text()
            baseline_law = _law_card(baseline)

            for case in cases:
                case_dir = mechanics_dir / case["id"]
                starter = (case_dir / case["starter"]).read_text()
                engine = (case_dir / case["engines"][0]).read_text()
                manifest = json.loads((case_dir / "case.json").read_text())
                self.assertEqual(_law_card(starter), baseline_law)
                self.assertEqual(case["parameters"]["sha256"], FROZEN_PARAMETERS_SHA256)
                self.assertEqual(manifest, case)
                self.assertFalse(case["physical_qualification"])
                self.assertIn("# Instrument-only LAW112 mechanics fixture", starter)
                self.assertIn("/ANIM/BRICK/TENS", engine)
                self.assertIn("/ANIM/BRICK/EPSP", engine)

            nodes = {
                1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0),
                3: (10.0, 10.0, 0.0), 4: (0.0, 10.0, 0.0),
                5: (0.0, 0.0, 10.0), 6: (10.0, 0.0, 10.0),
                7: (10.0, 10.0, 10.0), 8: (0.0, 10.0, 10.0),
            }
            shear = cases[0]
            shear_starter = (mechanics_dir / shear["id"] / shear["starter"]).read_text()
            groups = _node_groups(shear_starter)
            self.assertEqual(groups[1], [1, 2, 5, 6])
            self.assertEqual(groups[2], [3, 4, 7, 8])
            self.assertEqual(groups[3], list(range(1, 9)))
            self.assertIn(("011000", 3), _bcs(shear_starter))
            self.assertIn(("100000", 1), _bcs(shear_starter))
            impvel_index = shear_starter.splitlines().index("/IMPVEL/1/0")
            impvel_fields = shear_starter.splitlines()[impvel_index + 2].split()
            scale_fields = [float(value) for value in shear_starter.splitlines()[impvel_index + 3].split()]
            self.assertEqual(impvel_fields[1], "X")
            self.assertEqual(int(impvel_fields[4]), 2)
            target_displacement = scale_fields[1] * sum(
                (tb - ta) * (va + vb) * 0.5
                for (ta, va), (tb, vb) in zip(_function_points(shear_starter, 1),
                                              _function_points(shear_starter, 1)[1:])
            )
            gamma = target_displacement / 10.0
            for node_id, (x, y, z) in nodes.items():
                # The boundary data prescribe the exact affine simple-shear field
                # x' = x + gamma*y, y' = y, z' = z on all eight nodes.
                x_displacement = target_displacement if node_id in groups[2] else 0.0
                self.assertAlmostEqual(x_displacement, gamma * y, places=12)
                deformed = (x + x_displacement, y, z)
                self.assertAlmostEqual(deformed[0], x + gamma * y, places=12)
                self.assertAlmostEqual(deformed[1], y, places=12)
                self.assertAlmostEqual(deformed[2], z, places=12)
            shear_rigid_rows = []
            for point in nodes.values():
                for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                    shear_rigid_rows.append(_rigid_mode_row(point, direction))
            self.assertEqual(_rank(shear_rigid_rows), 6)
            self.assertAlmostEqual(
                shear["protocol"]["expected_elastic_shear_stress_xy_MPa"], 1.039
            )
            self.assertIn("S03 is the frozen 1e20 MPa inactive sentinel",
                          shear["protocol"]["yield_status"])

            plastic = cases[1]
            plastic_starter = (mechanics_dir / plastic["id"] / plastic["starter"]).read_text()
            self.assertIn("/IMPDISP/1/0", plastic_starter)
            self.assertIn("native MD tension then partial strain reversal", plastic_starter)
            self.assertIn("no force-free springback stage", plastic["protocol"]["scope"])

            rotation = cases[2]
            rotation_starter = (mechanics_dir / rotation["id"] / rotation["starter"]).read_text()
            rotation_lines = rotation_starter.splitlines()
            skew_index = rotation_lines.index("/SKEW/FIX/1/0")
            origin = tuple(float(value) for value in rotation_lines[skew_index + 2].split())
            skew_y = tuple(float(value) for value in rotation_lines[skew_index + 3].split())
            skew_z = tuple(float(value) for value in rotation_lines[skew_index + 4].split())
            self.assertEqual(origin, (5.0, 5.0, 5.0))
            self.assertEqual(skew_y, (0.0, 1.0, 0.0))
            self.assertEqual(skew_z, (0.0, 0.0, 1.0))
            imposed = {}
            for index, line in enumerate(rotation_lines):
                if line.startswith("/IMPDISP/"):
                    fields = rotation_lines[index + 2].split()
                    scales = [float(value) for value in rotation_lines[index + 3].split()]
                    self.assertEqual(int(fields[0]), 3)       # shared angular ramp
                    self.assertEqual(int(fields[2]), 1)       # centered skew
                    self.assertEqual(int(fields[4]), 1)       # all-node group
                    self.assertEqual(int(fields[6]), 1)       # cylindrical coordinates
                    imposed[fields[1]] = scales[1]
            self.assertEqual(imposed, {"Y": 0.1, "X": 0.0, "Z": 0.0})
            angle_curve = _function_points(rotation_starter, 3)
            self.assertEqual(angle_curve[0][1], 0.0)
            self.assertEqual(angle_curve[-1][1], 1.0)
            center = origin
            theta = imposed["Y"]
            rotated = {}
            for node_id, (x, y, z) in nodes.items():
                dx, dy = x - center[0], y - center[1]
                rotated[node_id] = (center[0] + dx * math.cos(theta) - dy * math.sin(theta),
                                    center[1] + dx * math.sin(theta) + dy * math.cos(theta), z)
            for a in nodes:
                for b in nodes:
                    initial_distance = math.dist(nodes[a], nodes[b])
                    rotated_distance = math.dist(rotated[a], rotated[b])
                    self.assertAlmostEqual(rotated_distance, initial_distance, places=12)
            cylindrical_rigid_rows = []
            for x, y, z in nodes.values():
                dx, dy = x - center[0], y - center[1]
                radius = math.hypot(dx, dy)
                radial = (dx / radius, dy / radius, 0.0)
                azimuthal = (-dy / radius, dx / radius, 0.0)
                for direction in (radial, azimuthal, (0.0, 0.0, 1.0)):
                    cylindrical_rigid_rows.append(_rigid_mode_row((x, y, z), direction))
            self.assertEqual(_rank(cylindrical_rigid_rows), 6)
            self.assertIn("not a dynamic rigid-body qualification", rotation["protocol"]["scope"])


if __name__ == "__main__":
    unittest.main()
