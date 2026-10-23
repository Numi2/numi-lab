#!/usr/bin/env python3
"""Focused tests for the source-derived LAW112 material-transfer exporter."""
import json
from pathlib import Path
import tempfile
import unittest

import cardboard_opencourant_material as material


FIT_SHA256 = "83f5778db797f0fb9996416848abb91c18f9102f4c3707aea9ea267b1eeb0d82"
YIELD_SHA256 = "ede6b47229eb02f1d1de278035c995eaa358919c55384398cbe33fe3de3c9f04"
MANIFEST_SHA256 = "6f3c556d6ffc97998d243601a6bfb34a1e8735026f06d6c0e8a8aec625a7d68f"
PARAMETERS_SHA256 = "46d33c7e6e4c796ad3285e269df03b1a0a6442dc70e6cb86de8a38ab5568527b"


def _card_rows(card: str) -> list[list[float]]:
    lines = card.splitlines()
    return [[float(token) for token in lines[index].split()] for index in range(7, 12)]


class SourceDerivedMaterialTransfer(unittest.TestCase):
    def test_direction_coefficients_recompute_from_fit_endpoint_and_anchor(self):
        expected = {
            ("liner", "MD"): (38.2, 4019.922690918769),
            ("liner", "CD"): (18.4, 609.0610696628671),
            ("medium", "MD"): (15.7, 1880.2071822828195),
            ("medium", "CD"): (8.06, 287.1312377908009),
        }
        for (role, axis), (s0_expected, h_expected) in expected.items():
            with self.subTest(role=role, axis=axis):
                record = material.source_record(role)
                branch = record["hardening_transfer"]["branches"][axis]
                coefficients = branch["LAW112_tensile_direction_parameters"]
                self.assertAlmostEqual(coefficients["S0_MPa"], s0_expected, places=10)
                self.assertEqual(coefficients["A_MPa"], 0.0)
                self.assertEqual(coefficients["B_per_plastic_strain"], 0.0)
                self.assertAlmostEqual(coefficients["C_MPa"], h_expected, places=9)
                self.assertAlmostEqual(branch["plastic_hardening_modulus_MPa"],
                                       h_expected, places=9)
                self.assertAlmostEqual(branch["nominal_transfer_prediction_MPa"],
                                       branch["digitized_fit_endpoint"]["stress_MPa"], places=10)
                low, high = branch["plastic_hardening_modulus_pixel_bound_MPa"]
                self.assertLess(low, h_expected)
                self.assertGreater(high, h_expected)

        for role, direction_index in (("liner", 0), ("medium", 1)):
            card = material.law_card(71, role)
            self.assertTrue(card.startswith("/MAT/LAW112/71\n"))
            rows = _card_rows(card)
            self.assertEqual([rows[0][1], rows[0][2]], [0.0, 0.0])
            self.assertAlmostEqual(rows[0][3], expected[(role, "MD")][1], places=7)
            self.assertEqual([rows[1][1], rows[1][2]], [0.0, 0.0])
            self.assertAlmostEqual(rows[1][3], expected[(role, "CD")][1], places=7)
            for row in rows[2:]:
                self.assertEqual(row[1:], [0.0, 0.0, 0.0])

    def test_card_keeps_current_elastic_and_density_proxy_fields(self):
        liner = _card_rows(material.law_card(1, "liner"))
        medium = _card_rows(material.law_card(1, "medium"))
        liner_header = material.law_card(1, "liner").splitlines()
        medium_header = material.law_card(1, "medium").splitlines()
        self.assertAlmostEqual(float(liner_header[2]), 7.66e-10)
        self.assertAlmostEqual(float(medium_header[2]), 6.79e-10)
        # S0 rows remain the Table 1 tensile and Table 3 compression mappings.
        self.assertEqual([row[0] for row in liner], [38.2, 18.4, 1e20, 22.92, 11.02])
        self.assertEqual([row[0] for row in medium], [15.7, 8.06, 1e20, 9.391, 4.836])
        record = material.source_record("medium")
        self.assertEqual(record["compression_mapping"]["hardening_coefficients"],
                         "A=B=C=0 explicit perfect-plastic placeholder, not an identified compression fit")
        self.assertTrue(any(item["parameter"] == "LAW112 plastic Poisson ratios"
                            for item in record["unidentified_parameters"]))
        self.assertTrue(any(item["parameter"] == "in_plane_and_out_of_plane_shear_strengths"
                            for item in record["unidentified_parameters"]))

    def test_source_hashes_and_rate_limit_are_retained_in_each_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            output.mkdir()
            cases = material.export_tensile_coupons(output)
            self.assertEqual(len(cases), 4)
            self.assertEqual({case["source_record"]["case"]["axis"] for case in cases}, {"MD", "CD"})
            self.assertEqual({case["source_record"]["case"]["material_fields"]["rho_i"]
                              for case in cases}, {7.66e-10, 6.79e-10})
            for case in cases:
                with self.subTest(case=case["id"]):
                    source_hashes = case["source_hashes_sha256"]
                    self.assertEqual(source_hashes["matter/examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv"], FIT_SHA256)
                    self.assertEqual(source_hashes["matter/examples/cardboard/source-data/haj_ali_2009_yield_anchors.csv"], YIELD_SHA256)
                    self.assertEqual(source_hashes["matter/examples/cardboard/manufacturing-source-manifest.json"], MANIFEST_SHA256)
                    self.assertEqual(source_hashes["matter/examples/cardboard/opencourant/paper-parameters.json"], PARAMETERS_SHA256)
                    self.assertIn("matter/examples/cardboard/source-data/catalog.json", source_hashes)
                    binding = case["source_record"]["hardening_transfer"]["pinned_source_bindings"]
                    self.assertEqual(binding["engine_law112_sha256"], material.ENGINE_LAW112_SHA256)
                    self.assertIn(f"OpenCourant/{material.SOURCE_COMMIT}/{binding['one_hex_qa_starter_path']}",
                                  source_hashes)
                    self.assertFalse(case["physical_qualification"])
                    case_record = case["source_record"]["case"]
                    self.assertEqual(case_record["rate_choice"]["strain_rate_s-1"],
                                     material.ENGINEERING_STRAIN_RATE_S_1)
                    self.assertAlmostEqual(case_record["expected_LAW112_endpoint_stress_MPa"],
                                           case_record["source_fit_endpoint_stress_MPa"], places=12)
                    self.assertIn("no rate-acceleration factor applied",
                                  case_record["rate_choice"]["status"])
                    self.assertIn("not stated", case["source_record"]["limitations"][3])
                    case_dir = output / case["id"]
                    saved = json.loads((case_dir / "case.json").read_text())
                    self.assertEqual(saved, case)
                    starter = (case_dir / case["starter"]).read_text()
                    engine = (case_dir / case["engines"][0]).read_text()
                    self.assertIn("/MAT/LAW112/1", starter)
                    self.assertIn("/TH/BRIC/1", starter)
                    self.assertIn("/TH/NODE/2", starter)
                    self.assertIn("/TH/TITLE", engine)
                    self.assertIn("/IMPL/NONL/01", engine)
                    self.assertIn("/IMPL/SOLV/01", engine)
                    self.assertIn("/ANIM/BRICK/TENS/STRESS\n", engine)
                    self.assertNotIn("/ANIM/BRICK/TENS\n", engine)
                    self.assertIn("/ANIM/BRICK/EPSP\n", engine)
                    self.assertEqual(case["source_record"]["case"]["history"]["animation"],
                                     ["BRICK/TENS/STRESS", "BRICK/EPSP", "BRICK/ENER", "VECT/DISP"])
                    self.assertNotIn("/DT/SCAL", engine)
                    self.assertNotIn("/MASS/", starter)

    def test_endpoint_hold_changes_only_drive_stop_not_physical_motion_or_run_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = root / "baseline"
            held_dir = root / "held"
            baseline_dir.mkdir()
            held_dir.mkdir()
            baseline_cases = material.export_tensile_coupons(baseline_dir)
            held_cases = material.export_tensile_coupons(held_dir, endpoint_hold_s=1.0)
            self.assertEqual(len(baseline_cases), 4)
            for baseline_case, held_case in zip(baseline_cases, held_cases):
                with self.subTest(case=baseline_case["id"]):
                    self.assertEqual(baseline_case["id"], held_case["id"])
                    self.assertEqual(baseline_case["expected_end_time_s"], held_case["expected_end_time_s"])
                    self.assertEqual(baseline_case["source_record"]["case"]["target_displacement_mm"],
                                     held_case["source_record"]["case"]["target_displacement_mm"])
                    self.assertEqual(baseline_case["source_record"]["case"]["rate_choice"],
                                     held_case["source_record"]["case"]["rate_choice"])
                    base_case_dir = baseline_dir / baseline_case["id"]
                    held_case_dir = held_dir / held_case["id"]
                    base_starter = (base_case_dir / baseline_case["starter"]).read_text().splitlines()
                    held_starter = (held_case_dir / held_case["starter"]).read_text().splitlines()
                    changed = [(old, new) for old, new in zip(base_starter, held_starter) if old != new]
                    self.assertEqual(len(changed), 2)
                    # The only native input edits are the flat /FUNCT end knot
                    # and /IMPVEL Tstop; the velocity scale is unchanged.
                    function_change = next((pair for pair in changed if len(pair[0].split()) == 2), None)
                    impvel_change = next((pair for pair in changed if len(pair[0].split()) == 4), None)
                    self.assertIsNotNone(function_change)
                    self.assertIsNotNone(impvel_change)
                    assert function_change is not None and impvel_change is not None
                    self.assertAlmostEqual(float(function_change[1].split()[0])
                                           - float(function_change[0].split()[0]), 1.0)
                    self.assertAlmostEqual(float(function_change[1].split()[1]), 1.0)
                    self.assertAlmostEqual(float(function_change[0].split()[1]), 1.0)
                    base_impvel = base_starter.index("/IMPVEL/1/0")
                    held_impvel = held_starter.index("/IMPVEL/1/0")
                    base_scales = [float(x) for x in base_starter[base_impvel + 3].split()]
                    held_scales = [float(x) for x in held_starter[held_impvel + 3].split()]
                    self.assertEqual(base_scales[:3], held_scales[:3])
                    self.assertAlmostEqual(held_scales[3] - base_scales[3], 1.0)
                    self.assertEqual((base_case_dir / baseline_case["engines"][0]).read_bytes(),
                                     (held_case_dir / held_case["engines"][0]).read_bytes())
                    window = held_case["source_record"]["case"]["imposed_motion_window"]
                    self.assertEqual(window["analysis_end_time_s"], held_case["expected_end_time_s"])
                    self.assertAlmostEqual(window["active_stop_time_s"] - window["analysis_end_time_s"], 1.0)
                    self.assertEqual(window["additional_active_time_s"], 1.0)
                    basis = window["native_source_basis"]
                    self.assertEqual(basis["commit"], material.SOURCE_COMMIT)
                    self.assertEqual(basis["reader_git_blob_sha1"],
                                     "4cdc9486967a46a7d65f7f4e0f203cdad7ff1862")
                    self.assertEqual(basis["engine_git_blob_sha1"],
                                     "da31dab9ed767d1633d36b1075c8d90eee9d8aa1")
                    self.assertIn("TT == STOPT remains active", basis["active_rule"])

    def test_rate_and_material_identifier_validation(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(rate=invalid), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                with self.assertRaises(ValueError):
                    material.export_tensile_coupons(output, strain_rate_s_1=invalid)
                self.assertEqual(list(output.iterdir()), [])
        for invalid_hold in (-1.0, float("nan"), float("inf")):
            with self.subTest(endpoint_hold=invalid_hold), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                with self.assertRaises(ValueError):
                    material.export_tensile_coupons(output, endpoint_hold_s=invalid_hold)
                self.assertEqual(list(output.iterdir()), [])
        for material_id in (0, -1, True, 1.5):
            with self.subTest(material_id=material_id), self.assertRaises(ValueError):
                material._law_card_for_fields(material_id, "liner", material._material_fields("liner")[1])


if __name__ == "__main__":
    unittest.main()
