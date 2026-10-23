from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

from matter.tools.cardboard_opencourant_overlay import (
    OVERLAY03_CASE_NAME,
    OverlayConfig,
    build_overlay_fixture,
    export_overlay_case,
    render_starter,
)


class CardboardOpenCourantOverlayTests(unittest.TestCase):
    def test_two_fixed_reference_substrates_and_independent_two_by_two_overlay(self) -> None:
        model = build_overlay_fixture()
        self.assertEqual(len(model.nodes), 34)
        self.assertEqual(len(model.substrate_elements), 2)
        self.assertEqual(len(model.cohesive_elements), 4)
        self.assertEqual(len(model.lower_overlay_nodes), 9)
        self.assertEqual(len(model.upper_overlay_nodes), 9)
        self.assertTrue(set(model.lower_overlay_nodes).isdisjoint(model.upper_overlay_nodes))
        substrate_nodes = {node for conn in model.substrate_elements.values() for node in conn}
        overlay_nodes = set(model.lower_overlay_nodes) | set(model.upper_overlay_nodes)
        self.assertTrue(substrate_nodes.isdisjoint(overlay_nodes))
        self.assertGreater(model.min_center_jacobian_mm3, 0.0)

        for a, b in zip(model.lower_overlay_nodes, model.upper_overlay_nodes):
            self.assertAlmostEqual(model.nodes[b][2] - model.nodes[a][2],
                                   model.config.initial_gap_mm)
        for element_id, conn in model.cohesive_elements.items():
            self.assertEqual(len(conn), 8)
            self.assertGreater(element_id, 100)

    def test_initial_rotation_is_rigid_and_does_not_change_substrate_reference(self) -> None:
        unrotated = build_overlay_fixture(OverlayConfig(initial_rotation_deg=0.0))
        rotated = build_overlay_fixture(OverlayConfig(initial_rotation_deg=23.0))
        for node_id, local_xyz in unrotated.nodes.items():
            rotated_xyz = rotated.nodes[node_id]
            x, y, z = local_xyz
            theta = 23.0 * 3.141592653589793 / 180.0
            self.assertAlmostEqual(rotated_xyz[0], x * math.cos(theta) - y * math.sin(theta))
            self.assertAlmostEqual(rotated_xyz[1], x * math.sin(theta) + y * math.cos(theta))
            self.assertAlmostEqual(rotated_xyz[2], z)
        self.assertEqual(rotated.substrate_elements, unrotated.substrate_elements)
        self.assertEqual(rotated.lower_master_face, unrotated.lower_master_face)
        self.assertEqual(rotated.upper_master_face, unrotated.upper_master_face)

    def test_type2_ties_and_remote_drives_are_explicit_in_starter(self) -> None:
        model = build_overlay_fixture()
        deck = render_starter(model)
        self.assertIn("/INTER/TYPE2/1\nLower 3x3 auxiliary", deck)
        self.assertIn("/INTER/TYPE2/2\nUpper 3x3 auxiliary", deck)
        self.assertIn("/SURF/SEG/21/0", deck)
        self.assertIn("/SURF/SEG/22/0", deck)
        self.assertIn("/GRBRIC/BRIC/7", deck)
        self.assertIn("0.0005", deck)
        self.assertIn("/IMPDISP/", deck)
        self.assertIn("/BCS/1\nFix lower remote grip only", deck)
        self.assertNotIn("/BCS/31", deck)
        self.assertNotIn("/BCS/32", deck)
        self.assertIn("cohesive state and energy density", deck)
        self.assertIn("overlay side-node motion for gap reconstruction", deck)

        lines = deck.splitlines()
        self.assertNotIn("/TH/", lines)
        type2_rows = [lines[lines.index(f"/INTER/TYPE2/{index}") + 2]
                      for index in (1, 2)]
        for row, (group_id, surface_id) in zip(type2_rows, ((31, 21), (32, 22))):
            self.assertEqual(len(row), 100)
            fields = [row[offset:offset + 10] for offset in range(0, 80, 10)]
            self.assertEqual([int(field) for field in fields[:7]],
                             [group_id, surface_id, 1000, 5, 0, 2, 0])
            self.assertEqual(fields[7], "          ")  # no secondary surface for node-group ties
            self.assertEqual(row[80:], f"{model.config.type2_search_distance_mm:20g}")

        for surface_id, expected in ((21, (1, 5, 6, 7, 8)),
                                     (22, (2, 9, 12, 11, 10))):
            row = lines[lines.index(f"/SURF/SEG/{surface_id}/0") + 2]
            self.assertEqual(len(row), 50)
            self.assertEqual(tuple(int(row[i:i + 10]) for i in range(0, 50, 10)), expected)

    def test_material_property_and_history_records_match_native_fixed_width_shapes(self) -> None:
        lines = render_starter(build_overlay_fixture()).splitlines()

        law117 = lines.index("/MAT/LAW117/1")
        self.assertEqual(len(lines[law117 + 2]), 20)  # density real
        self.assertEqual(len(lines[law117 + 3]), 80)  # two 20-column reals + four 10-column flags
        self.assertEqual(len(lines[law117 + 4]), 80)
        self.assertEqual(len(lines[law117 + 5]), 100)  # five 20-column reals/integers
        self.assertEqual(len(lines[law117 + 6]), 80)
        type43 = lines.index("/PROP/TYPE43/1")
        self.assertEqual(lines[type43 + 2], f"{1:10d}")

        law1 = lines.index("/MAT/LAW1/2")
        self.assertEqual(len(lines[law1 + 2]), 20)
        self.assertEqual(len(lines[law1 + 3]), 40)
        type14 = lines.index("/PROP/TYPE14/2")
        self.assertEqual([len(lines[type14 + offset]) for offset in range(2, 6)],
                         [100, 100, 100, 30])

        # Native /TH/NODE syntax is fixed-width variable names followed by
        # pairs of node ID and skew ID. /TH/BRIC uses fixed-width variables.
        node_hist = lines.index("/TH/NODE/101")
        self.assertEqual(lines[node_hist + 2], "REACX     REACY     REACZ     ")
        node_rows = lines[node_hist + 3:node_hist + 7]
        self.assertTrue(all(len(row) == 20 for row in node_rows))
        self.assertEqual([int(row[:10]) for row in node_rows], [1, 2, 3, 4])
        self.assertTrue(all(int(row[10:]) == 0 for row in node_rows))
        bric_hist = lines.index("/TH/BRIC/104")
        self.assertEqual(lines[bric_hist + 2],
                         "STRESS    STRAIN    IE        OFF       ")

        # The accepted native material/property record widths are also used
        # for substrate part/brick card identifiers and connectivity.
        brick = lines.index("/BRICK/1")
        self.assertEqual(len(lines[brick + 1]), 90)
        self.assertTrue(all(len(lines[i]) == 90 for i in range(brick + 1, brick + 5)))
        self.assertFalse(any(re.fullmatch(r"/TH/", line) for line in lines))

    def test_export_case_has_staged_decks_and_hash_bound_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overlay-case"
            case = export_overlay_case(output)
            self.assertEqual(case["starter"], "type2_cohesive_overlay_0000.rad")
            self.assertEqual(case["engines"], [
                "type2_cohesive_overlay_0001.rad",
                "type2_cohesive_overlay_0002.rad",
                "type2_cohesive_overlay_0003.rad",
            ])
            self.assertFalse(case["native_solver_run"])
            self.assertFalse(case["physical_qualification"])
            sidecar = (output / "mesh.json").read_bytes()
            self.assertEqual(case["mesh_sidecar_sha256"], hashlib.sha256(sidecar).hexdigest())
            mesh = json.loads(sidecar)
            self.assertEqual(mesh["quality"]["cohesive_hex8_count"], 4)
            self.assertEqual(mesh["quality"]["overlay_nodes_per_side"], 9)
            self.assertTrue(mesh["remote_grips"]["all_other_substrate_nodes_free"])
            self.assertEqual(mesh["stage_times_s"]["accepted_cohesive_birth"], 0.0005)
            self.assertTrue(mesh["synthetic_material_controls"]["auxiliary_mass_intentionally_zero"])
            self.assertEqual(mesh["synthetic_material_controls"]["law117_mass_calculation_flag"], 2)
            self.assertIn("/TFILE/4", (output / case["engines"][0]).read_text(encoding="utf-8"))
            stages = [(output / name).read_text(encoding="utf-8") for name in case["engines"]]
            self.assertIn("/IMPL/NONL/01", stages[0])
            self.assertIn("/IMPL/SOLV/02\n1 0 3 1e-10", stages[0])
            self.assertNotIn("/IMPL/SOLV/01", stages[0])
            self.assertIn("/DT/NODA/STOP", stages[1])
            self.assertIn("/IMPL/NONL/01", stages[2])
            self.assertIn("/IMPL/SOLV/02\n1 0 3 1e-10", stages[2])

    def test_invalid_stage_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage times"):
            build_overlay_fixture(OverlayConfig(birth_time_s=0.0004))

    def test_overlay03_uses_source_bound_areal_mass_analogue_not_gap_volume(self) -> None:
        config = OverlayConfig.hot_melt_mass_analogue()
        self.assertEqual(config.cohesive_mass_calculation_flag, 1)
        self.assertAlmostEqual(config.law117_density_input, 5.0e-11)
        self.assertAlmostEqual(config.overlay_mass_mg, 20.0)
        self.assertEqual(config.cohesive_density_source_sha256,
                         "db2f828fb1c18ffdc94c3dd5e0488cd966967cb1cb6c9a8cfc92b431a7f4cc5c")

        model = build_overlay_fixture(config)
        deck = render_starter(model, case_name=OVERLAY03_CASE_NAME)
        lines = deck.splitlines()
        self.assertIn("/BEGIN", lines)
        self.assertIn(OVERLAY03_CASE_NAME, lines)
        law117 = lines.index("/MAT/LAW117/1")
        self.assertAlmostEqual(float(lines[law117 + 2]), 5.0e-11)
        mass_fields = [lines[law117 + 3][i:i + 10] for i in (40, 50, 60, 70)]
        self.assertEqual([int(value) for value in mass_fields], [1, 1, 1, 0])
        self.assertIn("/INTER/TYPE2/1", lines)
        self.assertIn("/INTER/TYPE2/2", lines)

        audit = model.mesh_dict()
        self.assertEqual(audit["synthetic_material_controls"]["law117_mass_calculation_flag"], 1)
        self.assertEqual(audit["synthetic_material_controls"]["law117_density_input_unit"], "Mg/mm2")
        self.assertAlmostEqual(audit["adhesive_mass_analogue"]["expected_overlay_mass_mg"], 20.0)
        self.assertTrue(audit["adhesive_mass_analogue"]["not_starch_mass_or_fracture_calibration"])
        self.assertEqual(audit["mass_ledger"]["lower_substrate_H8_volume_mm3"], 800.0)
        self.assertEqual(audit["mass_ledger"]["upper_substrate_H8_volume_mm3"], 800.0)
        self.assertAlmostEqual(audit["mass_ledger"]["expected_starter_total_mass_Mg"], 0.00160002)
        self.assertAlmostEqual(audit["mass_ledger"]["expected_starter_total_mass_mg"], 1_600_020.0)
        self.assertEqual(audit["type2_mass_audit"]["spot_flag"], 5)
        self.assertTrue(audit["type2_mass_audit"]["no_arbitrary_mass_scale"])

    def test_overlay03_export_binds_analogue_source_and_mass_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overlay03"
            case = export_overlay_case(
                output,
                OverlayConfig.hot_melt_mass_analogue(),
                case_name=OVERLAY03_CASE_NAME,
            )
            self.assertEqual(case["id"], OVERLAY03_CASE_NAME)
            self.assertFalse(case["native_solver_run"])
            self.assertFalse(case["physical_qualification"])
            self.assertEqual(case["source_document"]["sha256"],
                             "db2f828fb1c18ffdc94c3dd5e0488cd966967cb1cb6c9a8cfc92b431a7f4cc5c")
            self.assertAlmostEqual(case["type2_mass_audit"]["expected_total_model_mass_change_mg"], 20.0)
            self.assertAlmostEqual(case["mass_ledger"]["expected_starter_total_mass_Mg"], 0.00160002)
            self.assertEqual(case["source_document"]["relative_path"],
                             "../sources/sikamelt-230-product-data-sheet.pdf")
            self.assertTrue(any("20 mg overlay" in gate for gate in case["gates"]))
            deck = (output / case["starter"]).read_text(encoding="utf-8")
            self.assertIn("/BEGIN\ntype2_cohesive_overlay_03\n", deck)
            self.assertIn("/RUN/type2_cohesive_overlay_03/1", (output / case["engines"][0]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
