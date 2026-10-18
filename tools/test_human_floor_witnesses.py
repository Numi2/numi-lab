#!/usr/bin/env python3
"""Focused fixtures for the retained floor-witness checker."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).with_name("check_human_floor_witnesses.py")
SPEC = importlib.util.spec_from_file_location("check_human_floor_witnesses", CHECKER)
CHECKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER_MODULE)


def make_run(path, regions, *, manifold=False, bad_lineage=False,
             bad_count=False, empty_bin=False, omit_region=None):
    path.mkdir(parents=True)
    witness_fields = [
        "accepted_step", "accepted_time_s", "pre_dynamics_step_index",
        "pre_dynamics_time_s", "region", "source_geometry_index",
        "point_query_index", "selected_skin_vertex_local",
        "selected_vertex_map_index", "region_error", "global_error",
        "nonfinite_jacobian_count", "selected_point_world_x_m",
        "selected_point_world_y_m", "selected_point_world_z_m",
    ]
    with (path / "resting-support-selected-skin-witness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=witness_fields)
        writer.writeheader()
        for step in (1, 2):
            for region in range(regions):
                if region == omit_region:
                    continue
                source = 1000 + region
                if manifold and region in (20, 32, 33, 34):
                    source = 1020
                if manifold and region in (30, 35, 36, 37):
                    source = 1030
                if bad_lineage and region == 34:
                    source = 9999
                writer.writerow({
                    "accepted_step": step,
                    "accepted_time_s": step * 0.0005,
                    "pre_dynamics_step_index": step - 1,
                    "pre_dynamics_time_s": (step - 1) * 0.0005,
                    "region": region,
                    "source_geometry_index": source,
                    "point_query_index": region,
                    "selected_skin_vertex_local": region,
                    "selected_vertex_map_index": 10000 + region,
                    "region_error": 0,
                    "global_error": 0,
                    "nonfinite_jacobian_count": 0,
                    "selected_point_world_x_m": 0,
                    "selected_point_world_y_m": 0,
                    "selected_point_world_z_m": 0 if region else 0.2,
                })
    with (path / "resting-surface-audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "min_skin_bed_gap_m", "nonfinite_skin_vertices",
            "mesh_zero_area_triangles", "mesh_nonfinite_area_triangles",
            "functional_geometry_status",
        ])
        writer.writeheader()
        writer.writerow({
            "min_skin_bed_gap_m": 0,
            "nonfinite_skin_vertices": 0,
            "mesh_zero_area_triangles": 0,
            "mesh_nonfinite_area_triangles": 0,
            "functional_geometry_status": 0,
        })
    if manifold:
        counts = [0] * 38
        counts[4] = 920
        for row in (20, 32, 33, 34, 30, 35, 36, 37):
            counts[row] = 10
        if empty_bin:
            counts[34] = 0
            counts[4] += 10
        total = sum(counts) + (1 if bad_count else 0)
        (path / "launcher.log").write_text(
            f"resting_support_geometry=full_registered_skin vertices={total} "
            f"regions=38 full_binding_count=86 region_sizes=[{','.join(map(str, counts))}]\n"
        )


class FloorWitnessCheckerTests(unittest.TestCase):
    def run_check(self, path, expected_regions):
        boundary = {0}
        return CHECKER_MODULE.check(path, boundary, 0.001, 2, expected_regions)

    def test_default_32_region_fixture_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 32)
            result = CHECKER_MODULE.check(run, {0}, 0.001, 2)
            self.assertTrue(result["passed"], result["failures"])

    def test_38_region_manifold_fixture_checks_allocation_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 38, manifold=True)
            result = self.run_check(run, 38)
            self.assertTrue(result["passed"], result["failures"])

    def test_38_region_fixture_rejects_wrong_source_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 38, manifold=True, bad_lineage=True)
            result = self.run_check(run, 38)
            self.assertFalse(result["passed"])
            self.assertTrue(any("source identity" in failure
                                for failure in result["failures"]))

    def test_38_region_fixture_rejects_incomplete_vertex_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 38, manifold=True, bad_count=True)
            result = self.run_check(run, 38)
            self.assertFalse(result["passed"])
            self.assertTrue(any("cover every registered skin vertex" in failure
                                for failure in result["failures"]))

    def test_38_region_fixture_rejects_empty_foot_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 38, manifold=True, empty_bin=True)
            result = self.run_check(run, 38)
            self.assertFalse(result["passed"])
            self.assertTrue(any("foot manifold source/derived regions are empty"
                                in failure for failure in result["failures"]))

    def test_38_region_fixture_rejects_missing_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            make_run(run, 38, manifold=True, omit_region=37)
            result = self.run_check(run, 38)
            self.assertFalse(result["passed"])
            self.assertTrue(any("region IDs do not match" in failure
                                for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
