#!/usr/bin/env python3
"""CPU-only checks for the source-data package and its fail-closed contracts."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cardboard_source_data as source_data


class SourceDataPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = source_data.load_catalog()

    def test_registered_csv_hashes_and_metadata_validate(self) -> None:
        self.assertEqual(source_data.validate_catalog(self.catalog), [])

    def test_plot_render_must_be_bound_to_registered_original_pdf(self) -> None:
        altered = json.loads(json.dumps(self.catalog))
        dataset = next(d for d in altered["datasets"] if d["id"] == "johnson_popil_pin_summary")
        dataset["digitization"]["render"]["source_pdf_sha256"] = "0" * 64
        errors = source_data.validate_catalog(altered)
        self.assertTrue(any("not bound to an associated source PDF" in error for error in errors))

    def test_pin_plot_pixels_cannot_be_silently_merged_with_article_source(self) -> None:
        force = next(d for d in self.catalog["datasets"]
                     if d["id"] == "johnson_popil_pin_force_displacement")
        self.assertEqual(force["source_ids"], ["johnson_popil_2016_author_presentation"])
        self.assertEqual(force["protocol_source_id"], "johnson_popil_2015_article")
        self.assertEqual(force["published_protocol"]["specimen_dimensions_mm"], [50.0, 160.0])
        self.assertEqual(force["published_protocol"]["platen_approach_mm_per_min"], 12.5)
        article = next(s for s in self.catalog["sources"]
                       if s["id"] == "johnson_popil_2015_article")
        self.assertEqual(article["source_text_extract"],
                         "johnson_popil_2015_pin_protocol_record.md")
        self.assertEqual(article["source_text_extract_sha256"],
                         source_data.sha256(source_data.DATA / article["source_text_extract"]))
        conflicts = {item["id"]: item for item in self.catalog["source_conflicts"]}
        self.assertEqual(conflicts["johnson_popil_pin_work_unit_conflict"]["status"],
                         "unresolved_dimensional_conflict")
        self.assertEqual(conflicts["johnson_popil_strength_range_across_artifacts"]
                         ["article_text_reported_strength_range_N_per_mm"], [0.17, 0.8])
        audit = source_data.package_audit(self.catalog)
        self.assertEqual(audit["pin_direct_plot_source_ids"],
                         ["johnson_popil_2016_author_presentation"])
        self.assertEqual(audit["pin_protocol_context_source_id"],
                         "johnson_popil_2015_article")
        self.assertEqual(audit["pin_summary_work_unit_conflict_status"],
                         "unresolved_dimensional_conflict")

        merged = json.loads(json.dumps(self.catalog))
        dataset = next(d for d in merged["datasets"]
                       if d["id"] == "johnson_popil_pin_force_displacement")
        dataset["source_ids"] = ["johnson_popil_2015_article",
                                  "johnson_popil_2016_author_presentation"]
        errors = source_data.validate_catalog(merged)
        self.assertTrue(any("direct plot source must be the 2016 presentation only" in error
                            for error in errors))

    def test_pin_engineering_geometry_is_hash_bound_and_not_source_exact(self) -> None:
        profile = next(p for p in self.catalog["fixture_profiles"]
                       if p["id"] == "johnson_popil_pin_adhesion_engineering_prototype")
        path = source_data.DATA / profile["file"]
        self.assertEqual(source_data.sha256(path), profile["sha256"])
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["publication_protocol"]["pin_count"], 6)
        geometry = data["engineering_prototype_geometry"]
        self.assertEqual(geometry["pin_diameter_mm"], 1.0)
        self.assertEqual(geometry["pin_pitch_across_width_mm"], 8.0)
        self.assertEqual(geometry["array_registration"], "centered_on_specimen_width")
        self.assertIn("not Emerson fixture dimensions", geometry["prohibited_interpretation"])

        altered = json.loads(json.dumps(self.catalog))
        altered["fixture_profiles"][0]["sha256"] = "0" * 64
        errors = source_data.validate_catalog(altered)
        self.assertTrue(any("pin fixture profile SHA256 mismatch" in error for error in errors))

    def test_nagasawa_fig9_is_one_grouped_envelope_not_ten_replicates(self) -> None:
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "nagasawa_2012_fig9_scored_bend_envelope")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 90)
        self.assertEqual([float(row["angle_deg"]) for row in rows], [float(a) for a in range(1, 91)])
        self.assertEqual(len({row["condition_group"] for row in rows}), 1)
        self.assertEqual(len({row["condition_id"] for row in rows}), 1)
        self.assertFalse({"trace_id", "sample_id", "specimen_id"} & set(rows[0]))
        self.assertNotIn("confidence_interval", dataset)
        self.assertNotIn("replicate_distribution", dataset)
        self.assertEqual(sum(row["lower_envelope_censored_at_zero_axis"] == "true" for row in rows), 34)
        self.assertTrue(all(row["observation_class"] == "digitized_group_envelope_estimate" for row in rows))
        self.assertTrue(all(row["source_pdf_sha256"] == "364dbc9c02d1a5bd6f7b1c2b309a2f3d502de791151b474d0989655f6f3f3e17" for row in rows))
        self.assertTrue(all(row["embedded_image_sha256"] == "fd26155edee7bd320ee47ad1ba00a9a9de4371803f6165961778f853e8d3c091" for row in rows))
        source = next(s for s in self.catalog["sources"] if s["id"] == "nagasawa_komiyama_2012")
        self.assertIn("outer edges", dataset["digitization"]["line_width_handling"])
        self.assertIn("two-pixel", dataset["digitization"]["line_width_handling"])
        self.assertEqual(source_data.validate_nagasawa_fig9_envelope(rows, dataset, {source["id"]: source}), [])
        ratio_note = dataset["score_gap_ratio_semantics"]
        self.assertEqual(ratio_note["nagasawa_article_sweep_0p09_to_0p40"], "main_creaser_roll_gap_over_measured_board_caliper")
        self.assertTrue(ratio_note["not_punch_indentation_ratio"])

    def test_nagasawa_fig9_group_split_and_pixels_fail_closed(self) -> None:
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "nagasawa_2012_fig9_scored_bend_envelope")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        bad_split = json.loads(json.dumps(dataset))
        bad_split["split"] = "pointwise_train_test"
        source = next(s for s in self.catalog["sources"] if s["id"] == "nagasawa_komiyama_2012")
        sources = {source["id"]: source}
        self.assertTrue(any("condition-group split" in error for error in source_data.validate_nagasawa_fig9_envelope(rows, bad_split, sources)))
        bad_pixel = [dict(row) for row in rows]
        bad_pixel[10]["moment_envelope_max_Nm_per_m"] = "0.0"
        self.assertTrue(any("high moment does not match its pixel" in error for error in source_data.validate_nagasawa_fig9_envelope(bad_pixel, dataset, sources)))
        invented_trace = [dict(row) for row in rows]
        invented_trace[0]["trace_id"] = "curve_1"
        self.assertTrue(any("individual trace/specimen identities" in error for error in source_data.validate_nagasawa_fig9_envelope(invented_trace, dataset, sources)))

    def test_nagasawa_fig9_extraction_reproduces_committed_csv_when_archives_exist(self) -> None:
        source = next(s for s in self.catalog["sources"] if s["id"] == "nagasawa_komiyama_2012")
        external = [Path(source["external_original_pdf"]), Path(source["embedded_figure_image"]["path"])]
        sips = shutil.which("sips")
        if not sips or not all(path.is_file() for path in external):
            self.skipTest("hash-bound J-STAGE PDF/image or macOS sips is not available on this host")
        script = Path(__file__).with_name("cardboard_nagasawa_fig9_digitize.py")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "reproduced.csv"
            result = subprocess.run(
                [sys.executable, str(script), "--source-pdf", str(external[0]),
                 "--figure-image", str(external[1]), "--output", str(output)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            committed = source_data.DATA / "nagasawa_2012_fig9_scored_bend_envelope.csv"
            self.assertEqual(output.read_bytes(), committed.read_bytes())

    def test_haj_ali_separates_source_fit_estimates_from_raw_curves(self) -> None:
        audit = source_data.package_audit(self.catalog)
        self.assertFalse(audit["haj_ali_full_hardening_curve_available"])
        self.assertIn("author-posted page-4 render", audit["haj_ali_curve_reason"])
        self.assertTrue(audit["haj_ali_bilinear_tension_fit_estimate_available"])
        self.assertIn("not raw experimental traces", audit["haj_ali_bilinear_tension_fit_reason"])
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "haj_ali_2009_yield_anchors")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        measured_tension = [r for r in rows if r["loading"] == "tension"]
        assumed_compression = [r for r in rows if r["loading"] == "compression"]
        self.assertEqual(len(measured_tension), 4)
        self.assertEqual(len(assumed_compression), 4)
        self.assertTrue(all(r["observation_class"] == "published_tensile_yield_anchor" for r in measured_tension))
        self.assertTrue(all(r["observation_class"] == "published_model_input_60pct_assumption" for r in assumed_compression))
        fit = next(d for d in self.catalog["datasets"] if d["id"] == "haj_ali_2009_bilinear_tension_fit_nodes")
        fit_path = source_data.DATA / fit["file"]
        self.assertEqual(source_data.sha256(fit_path), fit["sha256"])
        with fit_path.open(encoding="utf-8", newline="") as stream:
            fit_rows = list(csv.DictReader(stream))
        self.assertEqual(len(fit_rows), 12)
        self.assertEqual({r["series_id"] for r in fit_rows}, {
            "haj_ali_2009_liner_MD_bilinear_tension",
            "haj_ali_2009_liner_CD_bilinear_tension",
            "haj_ali_2009_medium_NSSC_MD_bilinear_tension",
            "haj_ali_2009_medium_NSSC_CD_bilinear_tension",
        })
        self.assertTrue(all(r["observation_class"] == "digitized_source_bilinear_fit_estimate_not_raw_measurement"
                            for r in fit_rows))
        self.assertEqual(sum(r["point_role"] == "digitized_bilinear_fit_endpoint" for r in fit_rows), 4)
        self.assertFalse(any("compression" in r["series_id"].lower() for r in fit_rows))
        for series_id in {r["series_id"] for r in fit_rows}:
            points = [r for r in fit_rows if r["series_id"] == series_id]
            anchor = next(r for r in points if r["point_role"] == "published_Table_1_tensile_yield_anchor")
            endpoint = next(r for r in points if r["point_role"] == "digitized_bilinear_fit_endpoint")
            e = float(endpoint["table2_elastic_modulus_MPa"])
            et = (float(endpoint["stress_MPa"]) - float(anchor["stress_MPa"])) / (
                float(endpoint["strain_mm_per_mm"]) - float(anchor["strain_mm_per_mm"]))
            h = e * et / (e - et)
            self.assertAlmostEqual(float(endpoint["postyield_tangent_MPa"]), et, places=12)
            self.assertAlmostEqual(float(endpoint["uniaxial_plastic_hardening_modulus_MPa"]), h, places=10)
            self.assertLess(float(endpoint["uniaxial_plastic_hardening_modulus_low_MPa"]), h)
            self.assertGreater(float(endpoint["uniaxial_plastic_hardening_modulus_high_MPa"]), h)
        liner_cd = next(r for r in fit_rows if r["series_id"] == "haj_ali_2009_liner_CD_bilinear_tension"
                        and r["point_role"] == "digitized_bilinear_fit_endpoint")
        self.assertEqual((int(liner_cd["x_pixel"]), int(liner_cd["y_pixel"])), (882, 49))

    def test_robertsson_eq18_energy_branch_conflict_is_exact_and_hash_bound(self) -> None:
        audit = source_data.package_audit(self.catalog)
        self.assertFalse(audit["robertsson_figure4_experimental_curve_available"])
        self.assertIn("unsupported application/vnd.ms-powerpoint", audit["robertsson_figure4_curve_reason"])
        self.assertIn("HTTP 403", audit["robertsson_figure4_curve_reason"])
        convention = audit["robertsson_eq18_energy_discrepancy"]
        self.assertEqual(convention["available_builder_conventions"], ["paper_absolute", "recentered"])
        self.assertTrue(convention["full_selected_energy_scales_damage_potential"])
        self.assertIsNone(convention["implicit_default"])
        self.assertEqual(convention["historical_regression_entry_point"],
                         "makeLegacyRecenteredMaterial")
        self.assertEqual(convention["required_source_verification_entry_point"],
                         "makePublishedReferenceMaterial(paper_absolute)")
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "robertsson_2023_eq18_compression_branch")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        compressed_limit = next(r for r in rows if r["branch_case"] == "compressed_side_limit_I13_to_1_minus")
        self.assertAlmostEqual(float(compressed_limit["paper_absolute_eq18_zd_term_MPa"]), -0.3332608695652174, places=14)
        self.assertEqual(float(compressed_limit["recentered_builder_zd_term_MPa"]), 0.0)
        self.assertAlmostEqual(float(compressed_limit["recentered_minus_paper_absolute_MPa"]),
                               0.3332608695652174, places=14)
        discrepancy = audit["robertsson_eq18_energy_discrepancy"]
        self.assertFalse(discrepancy["resolved"])
        self.assertTrue(discrepancy["conjugate_force_affected"])
        self.assertAlmostEqual(discrepancy["recentered_minus_paper_absolute_compression_constant_MPa"], 0.3332608695652174)
        self.assertEqual(len(discrepancy["source_extract_sha256"]), 64)
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "robertsson_2023_eq18_compression_branch")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        identity = next(r for r in rows if r["branch_case"] == "exact_identity_tension_branch")
        limit = next(r for r in rows if r["branch_case"] == "compressed_side_limit_I13_to_1_minus")
        self.assertAlmostEqual(float(identity["recentered_minus_paper_absolute_MPa"]), 0.0, places=14)
        self.assertAlmostEqual(float(limit["paper_absolute_eq18_zd_term_MPa"]), -0.3332608695652174, places=14)
        self.assertAlmostEqual(float(limit["recentered_builder_zd_term_MPa"]), 0.0, places=14)

    def test_robertsson_strength_hardening_is_bound_to_published_table_c2(self) -> None:
        audit = source_data.package_audit(self.catalog)
        self.assertTrue(audit["robertsson_strength_hardening_table_available"])
        self.assertIn("not current Kraft/NSSC", audit["robertsson_strength_hardening_scope"])
        dataset = next(d for d in self.catalog["datasets"]
                       if d["id"] == "robertsson_2023_strength_hardening")
        self.assertEqual(dataset["source_extract_sha256"],
                         next(s["source_text_extract_sha256"]
                              for s in self.catalog["sources"]
                              if s["id"] == "robertsson_2023"))
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        nu7 = next(row for row in rows if row["subsurface_nu"] == "7")
        self.assertEqual(nu7["hardening_law"], "linear")
        self.assertEqual(nu7["hardening_a_MPa"], "49.0")
        self.assertEqual(nu7["hardening_b"], "")
        nu9 = next(row for row in rows if row["subsurface_nu"] == "9")
        self.assertEqual(nu9["initial_yield_K0_MPa"], "2.51")
        self.assertEqual(nu9["hardening_a_MPa"], "")
        self.assertEqual(nu9["hardening_b"], "")

    def test_source_equation_extract_hash_is_verified(self) -> None:
        altered = json.loads(json.dumps(self.catalog))
        source = next(s for s in altered["sources"] if s["id"] == "robertsson_2023")
        source["source_text_extract_sha256"] = "0" * 64
        errors = source_data.validate_catalog(altered)
        self.assertTrue(any("source text extract SHA256 mismatch" in error for error in errors))

    def test_popil_equation_evaluation_is_not_table2_observation(self) -> None:
        rate_path = source_data.DATA / "popil_2012_ect_rate_fit.csv"
        table_path = source_data.DATA / "popil_2012_ect_table2.csv"
        with rate_path.open(encoding="utf-8") as stream:
            rate = list(csv.DictReader(stream))
        with table_path.open(encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        v125 = next(r for r in rate if float(r["speed_mm_min"]) == 12.5)
        commercial = next(r for r in table if r["series_id"] == "table2_commercial_standard_c")
        self.assertAlmostEqual(float(v125["ect_kN_m"]), 7.073612, places=6)
        self.assertAlmostEqual(float(commercial["T839_ECT_kN_m"]), 7.9, places=7)
        self.assertNotAlmostEqual(float(v125["ect_kN_m"]), float(commercial["T839_ECT_kN_m"]), places=2)
        self.assertEqual(v125["observation_class"], "published_equation_evaluation")

    def test_cohesive_gate_fails_closed(self) -> None:
        self.assertEqual(source_data.cohesive_status(self.catalog), 3)
        audit = source_data.package_audit(self.catalog)
        self.assertFalse(audit["cohesive_calibration_identified"])
        missing = audit["cohesive_parameters_unidentified"]
        self.assertIn("mode-I fracture energy GIc", missing)
        self.assertIn("mode-II fracture energy GIIc", missing)

    def test_pixel_transform_uses_image_downward_y_axis(self) -> None:
        axes = {"x_pixel_min": 10, "x_pixel_max": 110, "x_min": 11, "x_max": 17,
                "y_pixel_min": 20, "y_pixel_max": 120, "y_min": 0, "y_max": 600}
        self.assertEqual(source_data.pixel_to_units(10, 20, axes), (11.0, 600.0))
        self.assertEqual(source_data.pixel_to_units(110, 120, axes), (17.0, 0.0))
        self.assertEqual(source_data.pixel_to_units(60, 70, axes), (14.0, 300.0))
        with self.assertRaises(source_data.PackageError):
            source_data.pixel_to_units(111, 70, axes)
        with self.assertRaisesRegex(source_data.PackageError, "must be finite"):
            source_data.pixel_to_units(float("nan"), 70, axes)

    def test_acquisition_requires_public_pdf_and_external_destination(self) -> None:
        sources = {s["id"]: s for s in self.catalog["sources"]}
        url = source_data.select_public_pdf(sources["johnson_popil_2016_author_presentation"])
        self.assertTrue(url.endswith(".pdf"))
        with self.assertRaisesRegex(source_data.PackageError, "no registered public PDF"):
            source_data.select_public_pdf(sources["haj_ali_2009"])
        with self.assertRaisesRegex(source_data.PackageError, "outside the repository"):
            source_data.acquire_pdf(sources["johnson_popil_2016_author_presentation"], source_data.REPO / "matter" / "source.pdf")

    def test_digitization_requires_registered_original_hash(self) -> None:
        request = {
            "source_id": "johnson_popil_2016_author_presentation",
            "source_original_sha256": "not-the-registered-sha",
            "page": 15,
            "series_id": "test",
            "axes": {"x_pixel_min": 0, "x_pixel_max": 10, "x_min": 0, "x_max": 1,
                     "y_pixel_min": 0, "y_pixel_max": 10, "y_min": 0, "y_max": 1,
                     "x_unit": "mm", "y_unit": "N"},
            "uncertainty": {"x": 0.01, "y": 1},
            "points": [{"x_px": 1, "y_px": 9}, {"x_px": 9, "y_px": 1}],
        }
        with tempfile.TemporaryDirectory() as temp:
            request_path = Path(temp) / "request.json"
            output_path = Path(temp) / "result.csv"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(source_data.PackageError, "hash-verified original"):
                source_data.digitize_request(request_path, output_path, self.catalog)
            self.assertFalse(output_path.exists())

    def test_digitization_converts_hash_bound_points_and_labels_estimates(self) -> None:
        request = {
            "source_id": "johnson_popil_2016_author_presentation",
            "source_original_sha256": "4df7b5a4318fb12ea5afb31666e60dbaef7d0a0d4fbc9c337893ce4edcb3a4e5",
            "page_render_sha256": "7e7e1a645397d7d40db106d7698a444ea8436dcb1ad22b906b3bc16462903198",
            "page": 15,
            "series_id": "test_visible_branch",
            "axes": {"x_pixel_min": 10, "x_pixel_max": 110, "x_min": 11, "x_max": 17,
                     "y_pixel_min": 20, "y_pixel_max": 120, "y_min": 0, "y_max": 600,
                     "x_unit": "mm", "y_unit": "N"},
            "uncertainty": {"x": 0.02, "y": 5},
            "points": [{"x_px": 60, "y_px": 70}, {"x_px": 110, "y_px": 120}],
        }
        with tempfile.TemporaryDirectory() as temp:
            request_path = Path(temp) / "request.json"
            output_path = Path(temp) / "result.csv"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            source_data.digitize_request(request_path, output_path, self.catalog)
            with output_path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[0]["x"]), 14.0)
        self.assertAlmostEqual(float(rows[0]["y"]), 300.0)
        self.assertEqual(rows[0]["observation_class"], "digitized_estimate")
        self.assertEqual(rows[0]["source_original_sha256"], request["source_original_sha256"])
        self.assertEqual(rows[0]["source_render_sha256"], request["page_render_sha256"])

    def test_digitization_refuses_nonfinite_uncertainty_and_existing_output(self) -> None:
        request = {
            "source_id": "johnson_popil_2016_author_presentation",
            "source_original_sha256": "4df7b5a4318fb12ea5afb31666e60dbaef7d0a0d4fbc9c337893ce4edcb3a4e5",
            "page_render_sha256": "7e7e1a645397d7d40db106d7698a444ea8436dcb1ad22b906b3bc16462903198",
            "page": 15,
            "series_id": "test_visible_branch",
            "axes": {"x_pixel_min": 10, "x_pixel_max": 110, "x_min": 11, "x_max": 17,
                     "y_pixel_min": 20, "y_pixel_max": 120, "y_min": 0, "y_max": 600,
                     "x_unit": "mm", "y_unit": "N"},
            "uncertainty": {"x": float("nan"), "y": 5},
            "points": [{"x_px": 60, "y_px": 70}, {"x_px": 110, "y_px": 120}],
        }
        with tempfile.TemporaryDirectory() as temp:
            request_path = Path(temp) / "request.json"
            output_path = Path(temp) / "result.csv"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaisesRegex(source_data.PackageError, "uncertainties"):
                source_data.digitize_request(request_path, output_path, self.catalog)
            self.assertFalse(output_path.exists())
            request["uncertainty"] = {"x": 0.02, "y": 5}
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output_path.write_text("preserve existing evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(source_data.PackageError, "refusing to overwrite"):
                source_data.digitize_request(request_path, output_path, self.catalog)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserve existing evidence\n")

    def test_author_presentation_fit_is_descriptive_not_calibration(self) -> None:
        result = source_data.fit_pin_summary(self.catalog, "work")
        self.assertTrue(result["input_points_are_digitized_estimates"])
        self.assertIsNone(result["confidence_interval"])
        self.assertIn("unit unresolved", result["y"])
        self.assertIn("not admitted", result["prediction_admission"])
        self.assertGreater(result["fit"]["n"], 3)
        self.assertTrue(math.isfinite(result["fit"]["r_squared_descriptive"]))

    def test_pin_summary_preserves_work_unit_conflict(self) -> None:
        dataset = next(d for d in self.catalog["datasets"] if d["id"] == "johnson_popil_pin_summary")
        with (source_data.DATA / dataset["file"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r["pin_adhesion_work_unit_as_printed"] == "N-m" for r in rows))
        self.assertTrue(all(r["pin_adhesion_work_unit_status"] == "unresolved_conflict_with_slide15_integral_units" for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
