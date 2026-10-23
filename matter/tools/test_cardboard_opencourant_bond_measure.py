import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cardboard_opencourant_bond_measure import (
    LAW117_SOURCE_COMMIT,
    SUPPORTED_THTOCSV_BINARY_SHA256,
    MeasurementError,
    measure_run,
)


class BondMeasurementTests(unittest.TestCase):
    def _fixture(self, root: Path, *, mode="I", one_sample=False,
                 activation=False, converter_sha=SUPPORTED_THTOCSV_BINARY_SHA256):
        run = root / "runs" / "fixture-run"
        post = root / "post" / "fixture-run" / "T01"
        run.mkdir(parents=True)
        post.mkdir(parents=True)
        stem = "fixture"
        starter_name, engine_name = "fixture_0000.rad", "fixture_0001.rad"
        case_id = (
            "law117-dormant-activate-at-translated-geometry" if activation
            else "law117-mode-ii-shear-failure" if mode == "II"
            else "law117-mode-i-open-failure"
        )
        case = {
            "schema": "cardboard.opencourant.adhesion.case.v1",
            "id": case_id,
            "starter": starter_name,
            "engines": [engine_name],
            "expected_end_time_s": 0.0009,
            "parameters": {
                "motion_functions": [],
                **({"activation": {
                    "activation_time_s": 0.0005,
                    "expected_OFF_before": 1,
                    "expected_OFF_after": 0,
                }} if activation else {}),
            },
        }
        (run / starter_name).write_text(
            "/MAT/LAW117/1\nSynthetic fixture\n1e-6\n"
            "100 100 1 1 1 0\n0 0 10 10 0\n"
            "1.5 1.5 2 2 1\n1 1 0.3 0.3 3\n"
            "/PROP/TYPE43/1\nfixture\n1\n"
            "/NODE\n"
            "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
            "5 0 0 0\n6 1 0 0\n7 1 1 0\n8 0 1 0\n"
            "/TH/NODE/101\nbottom reaction history\nREACX REACY REACZ\n"
            "1 0\n2 0\n3 0\n4 0\n"
            "/TH/NODE/102\ntop separation history\nDX DY DZ VX VY VZ\n"
            "5 0\n6 0\n7 0\n8 0\n"
            "/TH/BRIC/103\ncohesive state\nSTRESS STRAIN IE OFF\n1\n/END\n"
        )
        (run / engine_name).write_text("/RUN/fixture/1\n0.0009\n")
        (run / "case.json").write_text(json.dumps(case))
        times = [0.0] if one_sample else [0.0, 0.0003, 0.0006, 0.0009]
        displacement = [0.0] if one_sample else [0.0, 0.05, 0.1, 0.15]
        stress = [0.0] if one_sample else [0.0, 5.0, 10.0, 7.5]
        off = [0.0] if activation else [1.0] * len(times)
        # The converter differentiates each reaction impulse over forward,
        # centered, or backward windows. Supply the matching signed force rows.
        force_avg = []
        for i in range(len(times)):
            if len(times) < 3:
                force_avg.append(0.0)
                continue
            left, right = ((0, 1) if i == 0 else
                           (len(times) - 2, len(times) - 1) if i == len(times) - 1
                           else (i - 1, i + 1))
            integrated = sum(
                0.5 * (stress[j] + stress[j + 1]) * (times[j + 1] - times[j])
                for j in range(left, right)
            )
            force_avg.append(integrated / (times[right] - times[left]))
        headers = ["time"]
        headers += [f"cohesive state 1 {name}" for name in
                    ("OFF", "SX", "SY", "SZ", "SXZ", "IE")]
        headers += [f"bottom reaction history {node} {axis}" for node in (1, 2, 3, 4)
                    for axis in ("REACX", "REACZ")]
        headers += [f"top separation history {node} {axis}" for node in (5, 6, 7, 8)
                    for axis in ("DX", "DZ")]
        rows = []
        for i, time in enumerate(times):
            is_shear = mode == "II"
            vals = [time, off[i], 0.0, 0.0,
                    0.0 if is_shear else stress[i],
                    stress[i] if is_shear else 0.0,
                    0.5 * stress[i] * displacement[i],
                    *([(-force_avg[i] / 4.0 if is_shear else 0.0),
                       (-force_avg[i] / 4.0 if not is_shear else 0.0)] * 4),
                    *([displacement[i] if is_shear else 0.0,
                       displacement[i] if not is_shear else 0.0] * 4)]
            rows.append(vals)
        csv_path = post / "probeT01.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        native = run / "fixtureT01"
        native.write_bytes(b"retained native history fixture")
        titles = run / "fixtureT01_TITLES"
        titles.write_bytes(b"retained title sidecar fixture")
        (post / "probeT01").write_bytes(native.read_bytes())
        (post / "probeT01_TITLES").write_bytes(titles.read_bytes())
        conversion = {
            "schema": "cardboard.opencourant.postprocess.v1",
            "receipt_created_after_conversion": True,
            "input_relative_path": "fixtureT01",
            "input_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "input_alias": "probeT01",
            "titles_sha256": hashlib.sha256(titles.read_bytes()).hexdigest(),
            "converter_sha256": converter_sha,
            "output": "probeT01.csv",
            "output_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "exit_code": 0,
        }
        (post / "conversion.json").write_text(json.dumps(conversion))
        input_map = {}
        for name in ("case.json", starter_name, engine_name):
            data = (run / name).read_bytes()
            input_map[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        run_record = {
            "schema": "cardboard.opencourant.run.v1",
            "execution_complete": True,
            "case_sha256": hashlib.sha256((run / "case.json").read_bytes()).hexdigest(),
            "source_commit": LAW117_SOURCE_COMMIT,
            "binary_sha256": {"engine": "a" * 64, "starter": "b" * 64},
            "runtime_identity_sha256": "c" * 64,
            "inputs": input_map,
            "outputs": {
                "fixtureT01": {"bytes": len(native.read_bytes()), "sha256": conversion["input_sha256"]},
                "fixtureT01_TITLES": {"bytes": len(titles.read_bytes()), "sha256": conversion["titles_sha256"]},
            },
            "commands": [{"exit_code": 0, "normal_termination": True,
                           "error_termination": False, "timed_out": False}],
        }
        (run / "run.json").write_text(json.dumps(run_record))
        return run, root / "post"

    def test_mode_i_uses_native_traction_and_actual_gap_for_work(self):
        with tempfile.TemporaryDirectory() as directory:
            run, post = self._fixture(Path(directory), mode="I")
            result = measure_run(run, post)
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["observed"]["native_cohesive_work"]["selected_work_N_mm"], 0.9375)
        self.assertAlmostEqual(result["observed"]["constitutive_crosscheck"]["max_abs_stress_error_MPa"], 0.0)
        self.assertLess(result["observed"]["support_reaction_force_windows"]["window_average_force_residual_max_abs_N"], 1e-12)
        self.assertEqual(result["observed"]["off_values"], [1])

    def test_mode_ii_uses_sxz_and_dx(self):
        with tempfile.TemporaryDirectory() as directory:
            run, post = self._fixture(Path(directory), mode="II")
            result = measure_run(run, post)
        self.assertEqual(result["observed"]["constitutive_crosscheck"]["observed_traction_channel"], "SXZ")
        self.assertEqual(result["observed"]["constitutive_crosscheck"]["measured_separation_component"], "DX")
        self.assertAlmostEqual(result["observed"]["native_cohesive_work"]["selected_work_N_mm"], 0.9375)

    def test_single_sample_activation_fails_coverage_and_uses_off_one_as_active(self):
        with tempfile.TemporaryDirectory() as directory:
            run, post = self._fixture(Path(directory), one_sample=True, activation=True)
            result = measure_run(run, post)
        self.assertEqual(result["status"], "coverage_failed")
        self.assertFalse(result["coverage"]["coverage_ok"])
        self.assertEqual(result["activation"]["status"], "coverage_failed")
        self.assertEqual(result["activation"]["OFF_observed_values"], [0.0])
        self.assertTrue(result["activation"]["legacy_case_OFF_metadata_conflicts_with_native_semantics"])

    def test_unknown_converter_binary_is_not_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            run, post = self._fixture(Path(directory), converter_sha="0" * 64)
            with self.assertRaisesRegex(MeasurementError, "unsupported th_to_csv"):
                measure_run(run, post)

    def test_bond_normal_axis_x_uses_sx_dx_and_reacx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "rotated-frame"
            post = root / "post" / "rotated-frame" / "T01"
            run.mkdir(parents=True)
            post.mkdir(parents=True)
            starter_name, engine_name = "rotated_0000.rad", "rotated_0001.rad"
            case = {
                "schema": "cardboard.opencourant.bond.v1",
                "id": "rotated-frame-normal-x",
                "starter": starter_name,
                "engines": [engine_name],
                "expected_end_time_s": 0.0009,
                "normal_axis": "X",
                "activation": False,
                "prediction": {"final_force_N": 10.0},
            }
            starter = (
                "/MAT/LAW117/1\nSynthetic fixture\n1e-6\n"
                "100 100 1 1 1 0\n0 0 10 10 0\n"
                "1.5 1.5 2 2 1\n1 1 0.3 0.3 3\n"
                "/PROP/TYPE43/1\nfixture\n1\n"
                "/NODE\n"
                "1 0 0 0\n2 0 1 0\n3 0 1 1\n4 0 0 1\n"
                "5 0 0 0\n6 0 1 0\n7 0 1 1\n8 0 0 1\n"
                "9 0 0 0\n10 0 1 0\n11 0 1 1\n12 0 0 1\n"
                "/TH/NODE/101\nbottom reaction history\nREACX REACZ\n"
                "1 0\n2 0\n3 0\n4 0\n"
                "/TH/NODE/102\ntop separation history\nDX DZ\n"
                "5 0\n6 0\n7 0\n8 0\n"
                "/TH/BRIC/103\ncohesive state\nSTRESS STRAIN IE OFF\n1\n"
                "/TH/NODE/104\nouter gripper history\nDX REACX\n"
                "9 0\n10 0\n11 0\n12 0\n"
                "/TH/BRIC/105\nsolid adherend state\nSTRESS IE\n2\n/END\n"
            )
            (run / starter_name).write_text(starter)
            (run / engine_name).write_text("/RUN/fixture/1\n0.0009\n")
            case_path = run / "case.json"
            case_path.write_text(json.dumps(case))
            times = [0.0, 0.0003, 0.0006, 0.0009]
            displacement = [0.0, 0.05, 0.1, 0.15]
            stress = [0.0, 5.0, 10.0, 7.5]
            headers = ["time"]
            headers += [f"cohesive state 1 {name}" for name in
                        ("OFF", "SX", "SY", "SZ", "SXZ", "IE")]
            headers += [f"bottom reaction history {node} {axis}"
                        for node in (1, 2, 3, 4) for axis in ("REACX", "REACZ")]
            headers += [f"top separation history {node} {axis}"
                        for node in (5, 6, 7, 8) for axis in ("DX", "DZ")]
            headers += [f"outer gripper history {node} {axis}"
                        for node in (9, 10, 11, 12) for axis in ("DX", "REACX")]
            headers += [f"solid adherend state 2 {axis}" for axis in ("SX", "SY", "SZ", "IE")]
            force_avg = []
            for i in range(len(times)):
                left, right = ((0, 1) if i == 0 else
                               (len(times) - 2, len(times) - 1) if i == len(times) - 1
                               else (i - 1, i + 1))
                integrated = sum(
                    0.5 * (stress[j] + stress[j + 1]) * (times[j + 1] - times[j])
                    for j in range(left, right)
                )
                force_avg.append(integrated / (times[right] - times[left]))
            rows = []
            for i, time in enumerate(times):
                rows.append([
                    time, 1.0, stress[i], 0.0, 50.0 * stress[i], 0.0,
                    0.5 * stress[i] * displacement[i],
                    *([-force_avg[i] / 4.0, 0.0] * 4),
                    *([displacement[i], 10.0 * displacement[i]] * 4),
                    *([1.2 * displacement[i], 0.0] * 4),
                    0.0, 0.0, 50.0 * stress[i], 0.0,
                ])
            csv_path = post / "probeT01.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(headers)
                writer.writerows(rows)
            native, titles = run / "fixtureT01", run / "fixtureT01_TITLES"
            native.write_bytes(b"rotated native fixture")
            titles.write_bytes(b"rotated title fixture")
            (post / "probeT01").write_bytes(native.read_bytes())
            (post / "probeT01_TITLES").write_bytes(titles.read_bytes())
            conversion = {
                "schema": "cardboard.opencourant.postprocess.v1",
                "receipt_created_after_conversion": True,
                "input_relative_path": "fixtureT01",
                "input_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                "input_alias": "probeT01",
                "titles_sha256": hashlib.sha256(titles.read_bytes()).hexdigest(),
                "converter_sha256": SUPPORTED_THTOCSV_BINARY_SHA256,
                "output": "probeT01.csv",
                "output_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "exit_code": 0,
            }
            (post / "conversion.json").write_text(json.dumps(conversion))
            inputs = {}
            for name in ("case.json", starter_name, engine_name):
                data = (run / name).read_bytes()
                inputs[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            receipt = {
                "schema": "cardboard.opencourant.run.v1",
                "execution_complete": True,
                "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
                "source_commit": LAW117_SOURCE_COMMIT,
                "binary_sha256": {"engine": "a" * 64, "starter": "b" * 64},
                "runtime_identity_sha256": "c" * 64,
                "inputs": inputs,
                "outputs": {
                    "fixtureT01": {"bytes": len(native.read_bytes()), "sha256": conversion["input_sha256"]},
                    "fixtureT01_TITLES": {"bytes": len(titles.read_bytes()), "sha256": conversion["titles_sha256"]},
                },
                "commands": [{"exit_code": 0, "normal_termination": True,
                               "error_termination": False, "timed_out": False}],
            }
            (run / "run.json").write_text(json.dumps(receipt))
            result = measure_run(run, root / "post")
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["bond_interface"]["normal_axis"], "X")
        self.assertAlmostEqual(result["observed"]["native_cohesive_work"]["selected_work_N_mm"], 0.9375)
        self.assertAlmostEqual(result["observed"]["traction_max_MPa"], 10.0)
        self.assertEqual(result["observed"]["traction_component"], "SX")
        self.assertLess(result["observed"]["support_reaction_force_windows"]["window_average_force_residual_max_abs_N"], 1e-12)
        self.assertAlmostEqual(
            result["observed"]["support_reaction_force_windows"]["endpoint_force_vs_endpoint_traction_N"],
            1.25,
        )


if __name__ == "__main__":
    unittest.main()
