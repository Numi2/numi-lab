import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cardboard_opencourant_measure import (
    MeasurementError,
    SUPPORTED_THTOCSV_BINARY_SHA256,
    measure_run,
)


class ImplicitMeasurementTests(unittest.TestCase):
    TIMES = [0.0, 0.0001, 0.000201, 0.00030301, 0.0004060401,
             0.0005101005, 0.0006152015, 0.0007203025,
             0.0008264545, 0.0009326065, 0.001]

    def _stencil_force(self, index):
        times = self.TIMES
        if index == 0:
            left, right = times[0], times[1]
        elif index == len(times) - 1:
            left, right = times[-2], times[-1]
        else:
            left, right = times[index - 1], times[index + 1]
        return (left + right) / (2.0 * 0.001)

    def _fixture(self, root: Path, *, generic_reaction=False, nonmonotonic=False,
                 execution_complete=True, backward_everywhere=False, reversed_reaction=False,
                 converter_sha=SUPPORTED_THTOCSV_BINARY_SHA256):
        evidence = root / "evidence"
        run_dir = evidence / "runs" / "implicit-elastic-nonlinear-force-control-instrumented"
        post_dir = evidence / "post" / "implicit-elastic-shortname"
        run_dir.mkdir(parents=True)
        post_dir.mkdir(parents=True)
        stem = "implicit_elastic_nonlinear_force_control_instrumented"
        starter = f"{stem}_0000.rad"
        case = {
            "id": "implicit-elastic-nonlinear-force-control-instrumented",
            "starter": starter,
            "expected_end_time_s": 0.001,
            "prediction": {
                "total_applied_force_N": 1.0,
                "mean_upper_DZ_mm": 0.01,
                "sum_bottom_reaction_magnitude_N": 1.0,
                "elastic_work_N_mm": 0.005,
            },
            "gates": {
                "relative_displacement_error_max": 0.01,
                "relative_reaction_error_max": 0.01,
            },
            "physical_qualification": False,
            "native_outputs": {
                "time_history": {
                    "force_semantics": "REACZ is stored as impulse in T01; use generated T01_TITLES sidecar with official th_to_csv to derive reaction force",
                    "force_channel": "/TH/NODE/101 REACZ on bottom nodes 1-4; sum the signed normal reactions",
                    "displacement_channel": "/TH/NODE/102 DZ on top nodes 5-8; average the four values",
                    "energy_channel": "/TH/BRIC/103 IE for brick 1, in N mm",
                },
                "history_interval_s": 0.00001,
            },
        }
        (run_dir / "case.json").write_text(json.dumps(case))
        (run_dir / starter).write_text(
            "/GRNOD/NODE/1\nbottom face\n1 2 3 4\n"
            "/GRNOD/NODE/2\ntop face\n5 6 7 8\n"
            "/FUNCT/1\nlinear load ramp\n0 0\n0.001 1\n"
            "/TH/NODE/101\nbottom reaction history\nREACX REACY REACZ\n"
            "1 0\n2 0\n3 0\n4 0\n"
            "/TH/NODE/102\ntop separation history\nDX DY DZ VX VY VZ\n"
            "5 0\n6 0\n7 0\n8 0\n"
            "/TH/BRIC/103\ncohesive state\nSTRESS STRAIN IE OFF\n1\n"
            "/CLOAD/1\nUniform normal load; 1 N total\n"
            "         1         Z         0         0         2                             1                0.25\n"
            "/END\n")
        t01 = run_dir / f"{stem}T01"
        t01.write_bytes(b"fixture-native-history\n")
        titles_path = run_dir / f"{stem}T01_TITLES"
        titles_path.write_text("fixture titled history\n")

        headers = ["time"]
        if generic_reaction:
            headers.extend([f"bottom reaction history {node} var {node + 20}"
                            for node in range(1, 5)])
        else:
            headers.extend([f"bottom reaction history {node} REACZ"
                            for node in range(1, 5)])
        headers.extend([f"top separation history {node} DZ" for node in range(5, 9)])
        headers.append("cohesive state 1 IE")
        times = self.TIMES.copy()
        if nonmonotonic:
            times[5] = times[4]
        csv_path = post_dir / "probeT01.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for index, time_s in enumerate(times):
                row = [time_s]
                if backward_everywhere:
                    left = times[max(0, index - 1)]
                    right = times[index]
                    force_magnitude = ((left + right) / (2 * 0.001)
                                       if index else (times[0] + times[1]) / (2 * 0.001))
                else:
                    force_magnitude = self._stencil_force(index)
                node_reaction = (0.25 if reversed_reaction else -0.25) * force_magnitude
                row.extend([float(node_reaction)] * 4)
                row.extend([time_s / 0.001 * 0.01] * 4)
                row.append(time_s / 0.001 * 0.005)
                writer.writerow(row)
        alias_t01 = post_dir / "probeT01"
        alias_t01.write_bytes(t01.read_bytes())
        alias_titles = Path(str(alias_t01) + "_TITLES")
        alias_titles.write_bytes(titles_path.read_bytes())
        conversion = {
            "schema": "cardboard.opencourant.postprocess.v1",
            "receipt_created_after_conversion": True,
            "input_relative_path": f"runs/{run_dir.name}/{t01.name}",
            "input_sha256": hashlib.sha256(t01.read_bytes()).hexdigest(),
            "input_alias": "probeT01",
            "titles_sha256": hashlib.sha256(titles_path.read_bytes()).hexdigest(),
            "converter_sha256": converter_sha,
            "exit_code": 0,
            "output": csv_path.name,
            "output_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "argv": ["/runtime/th_to_csv_linuxa64", "probeT01"],
        }
        receipt_path = post_dir / "conversion.json"
        receipt_path.write_text(json.dumps(conversion))
        (run_dir / "run.json").write_text(json.dumps({
            "schema": "cardboard.opencourant.run.v1",
            "execution_complete": execution_complete,
            "commands": [{"exit_code": 0, "normal_termination": True,
                          "error_termination": False, "timed_out": False}],
            "outputs": {
                t01.name: {"sha256": hashlib.sha256(t01.read_bytes()).hexdigest()},
                titles_path.name: {"sha256": hashlib.sha256(titles_path.read_bytes()).hexdigest()},
            },
        }))
        return run_dir, receipt_path

    def test_official_stencil_curve_passes_and_endpoint_comparison_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory))
            result = measure_run(run_dir, conversion_receipt_path=receipt)
        self.assertEqual(result["overall_status"], "pass")
        self.assertFalse(result["physical_qualification"])
        self.assertAlmostEqual(result["displacement"]["mean_upper_DZ_mm"], 0.01)
        self.assertAlmostEqual(result["energy"]["brick_1_IE_final_N_mm"], 0.005)
        self.assertAlmostEqual(result["reaction"]["final_magnitude_N"], 0.96630325)
        self.assertLessEqual(result["reaction"]["gate"]["maximum_sample_relative_error"], 1e-12)
        self.assertEqual(result["reaction"]["gate"]["samples"][1]["method"], "centered")
        self.assertEqual(result["reaction"]["gate"]["samples"][-1]["method"], "backward")
        endpoint = result["reaction"]["endpoint_target_diagnostic"]
        self.assertFalse(endpoint["gate_applicable"])
        self.assertGreater(endpoint["relative_error"], 0.03)
        self.assertAlmostEqual(result["external_work_crosscheck"]["value_N_mm"], 0.005)
        self.assertFalse(result["reaction_work_quadrature_diagnostic"]["gate_applicable"])

    def test_interior_samples_are_not_compared_as_backward_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), backward_everywhere=True)
            result = measure_run(run_dir, conversion_receipt_path=receipt)
        self.assertEqual(result["reaction"]["status"], "fail")
        self.assertFalse(result["reaction"]["gate"]["pass"])
        self.assertEqual(result["overall_status"], "fail")

    def test_generic_var_columns_are_never_treated_as_force(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), generic_reaction=True)
            result = measure_run(run_dir, conversion_receipt_path=receipt)
        self.assertEqual(result["overall_status"], "inconclusive")
        self.assertIsNone(result["reaction"]["source"])
        self.assertIsNone(result["reaction"]["gate"])
        self.assertIsNotNone(result["external_work_crosscheck"]["value_N_mm"])
        self.assertTrue(result["displacement"]["gate"]["pass"])
        self.assertTrue(result["energy"]["gate"]["pass"])

    def test_reaction_must_oppose_the_prescribed_load(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), reversed_reaction=True)
            result = measure_run(run_dir, conversion_receipt_path=receipt)
        self.assertEqual(result["reaction"]["status"], "fail")
        self.assertEqual(result["overall_status"], "fail")

    def test_nonmonotonic_timebase_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), nonmonotonic=True)
            with self.assertRaisesRegex(MeasurementError, "strictly increasing"):
                measure_run(run_dir, conversion_receipt_path=receipt)

    def test_failed_native_run_cannot_become_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), execution_complete=False)
            with self.assertRaisesRegex(MeasurementError, "execution_complete"):
                measure_run(run_dir, conversion_receipt_path=receipt)

    def test_converter_binary_must_match_the_verified_stencil_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, receipt = self._fixture(Path(directory), converter_sha="a" * 64)
            with self.assertRaisesRegex(MeasurementError, "unsupported th_to_csv binary"):
                measure_run(run_dir, conversion_receipt_path=receipt)


if __name__ == "__main__":
    unittest.main()
