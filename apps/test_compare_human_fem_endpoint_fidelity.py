import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("compare_human_fem_endpoint_fidelity.py")
spec = importlib.util.spec_from_file_location("endpoint_fidelity", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def make_log(*, stretch=1.1, second_stretch=None, reaction=10.0, dt="0.001", mass="1.0",
             contact_available="1", residual_available="1", reference_mode=0,
             region_ids=(7,), free_available="1", tag2_available="0",
             tag2_nodes=0, tag2_force=0.0):
    flags = {
        "bulk_reaction_available": "1",
        "contact_force_available": contact_available,
        "mechanical_residual_available": residual_available,
        "free_internal_force_available": free_available,
        "tag2_attachment_force_available": tag2_available,
    }
    stretches = [stretch] if second_stretch is None else [stretch, second_stretch]
    tets = []
    rest_sets = []
    for tet_index in range(len(stretches)):
        offset = tet_index * 3.0
        rest_sets.append(((offset, 0.0, 0.0), (offset + 1.0, 0.0, 0.0),
                          (offset, 1.0, 0.0), (offset, 0.0, 1.0)))
        base = tet_index * 4
        tets.append((base, base + 1, base + 2, base + 3))
    node_count = 4 * len(stretches)
    label_count = min(len(region_ids), len(stretches))
    header = ("resting_FEM_endpoint_capture=begin version=1 steps=1 dt_s=" + dt +
              " endpoint_time_s=" + dt + " runtime_dt_s=" + dt +
              " timestep_multiplier=1 timestep_divisor=1 completed_microsteps=1 "
              " coordinate_unit=m mass_unit=kg force_unit=N time_unit=s strain_unit=1 "
              " mechanical_residual_unit=N_s " +
              " ".join(f"{k}={v}" for k, v in flags.items()) +
              f" source_region_label_source=registered_skin_fat_donor_body_id source_region_labeled_nodes={label_count * 4} source_region_total_nodes={node_count}"
              f" source_region_labeled_tetrahedra={label_count} source_region_total_tetrahedra={len(stretches)}"
              f" force_channels_scope=FEM_bulk_contact objects=1 nodes={node_count} tetrahedra={len(stretches)} reactions=1")
    lines = [header]
    for tet_index, (rest, tet_stretch) in enumerate(zip(rest_sets, stretches)):
        for within_tet, point in enumerate(rest):
            i = tet_index * 4 + within_tet
            accepted = list(point)
            accepted[0] = point[0] + (tet_stretch - 1.0) * (1.0 if within_tet == 1 else 0.0)
            rx = reaction if i == 0 else 0.0
            hi = tuple(accepted[axis] - point[axis] for axis in range(3))
            lo = (-1e-10, 1e-10, 0.0) if reference_mode else (0.0, 0.0, 0.0)
            node_line = (
                "resting_FEM_node,object=Human/test-region,local={i},global={i},"
                "anchor_body=4294967295,constraint_tag={tag},"
                "initial_x={x0},initial_y={y1},initial_z={z2},"
                "accepted_x={a0},accepted_y={a1},accepted_z={a2},"
                "delta_x={dx},delta_y={dy},delta_z={dz},vx=0,vy=0,vz=0,"
                "mass_kg={mass},initial_mass_kg={mass},inverse_mass=1,"
                "rest_x={x0},rest_y={y1},rest_z={z2},reference_mode={mode},"
                "reference_u_hi_x={uh0},reference_u_hi_y={uh1},reference_u_hi_z={uh2},"
                "reference_u_lo_x={ul0},reference_u_lo_y={ul1},reference_u_lo_z={ul2},"
                "bulk_reaction_x_N={rx},bulk_reaction_y_N=0,bulk_reaction_z_N=0,"
                "contact_force_x_N=0,contact_force_y_N=0,contact_force_z_N=0,"
                "free_internal_force_x_N=0,free_internal_force_y_N=0,free_internal_force_z_N=0,"
                "tag2_attachment_force_x_N={tag2x},tag2_attachment_force_y_N=0,tag2_attachment_force_z_N=0,"
                "residual_impulse_x_Ns=0,residual_impulse_y_Ns=0,residual_impulse_z_Ns=0,"
                "residual_pressure_w=0".format(
                    i=i, x0=point[0], y1=point[1], z2=point[2],
                    a0=accepted[0], a1=accepted[1], a2=accepted[2],
                    dx=accepted[0]-point[0], dy=accepted[1]-point[1], dz=accepted[2]-point[2],
                    mass=mass, rx=rx, mode=reference_mode,
                    tag=2 if i < tag2_nodes else 0,
                    tag2x=tag2_force if (i < tag2_nodes or (tag2_nodes == 0 and i == 0)) else 0.0,
                    uh0=hi[0], uh1=hi[1], uh2=hi[2], ul0=lo[0], ul1=lo[1], ul2=lo[2]))
            if tet_index < label_count:
                node_line += (",anatomical_region_id={region},"
                              "anatomical_region_source=registered_skin_fat_donor_body").format(
                                  region=region_ids[tet_index])
            lines.append(node_line)
        base = tet_index * 4
        tet_line = ("resting_FEM_tet,object=Human/test-region,local={local},global={local},"
                    "n0={n0},n1={n1},n2={n2},n3={n3},J={stretch}").format(
                        local=tet_index, n0=base, n1=base+1, n2=base+2, n3=base+3,
                        stretch=tet_stretch)
        if tet_index < label_count:
            tet_line += (",anatomical_region_id={region},"
                         "anatomical_region_source=registered_skin_fat_donor_body").format(
                             region=region_ids[tet_index])
        lines.append(tet_line)
    lines.append("resting_FEM_anchor_reaction,step=0,x_N=0,y_N=0,z_N=0,magnitude_N=0")
    lines.append("resting_FEM_endpoint_capture=end")
    return "\n".join(lines) + "\n"


class EndpointFidelityTests(unittest.TestCase):
    def compare_logs(self, full_text, reduced_text):
        with tempfile.TemporaryDirectory() as temp:
            full = Path(temp) / "full.log"
            reduced = Path(temp) / "reduced.log"
            full.write_text(full_text)
            reduced.write_text(reduced_text)
            return module.compare(full, reduced)

    def test_identical_endpoint_passes(self):
        report = self.compare_logs(make_log(), make_log())
        self.assertTrue(report["gates"]["measured_channel_fidelity_pass"])
        self.assertEqual(report["green_strain_fidelity"]["gate"], "pass")
        self.assertEqual(report["force_fidelity"]["gate"], "pass")
        json.dumps(report, allow_nan=False)

    def test_force_error_over_one_percent_fails(self):
        report = self.compare_logs(make_log(reaction=10.0), make_log(reaction=10.2))
        self.assertEqual(report["force_fidelity"]["gate"], "fail")
        self.assertEqual(report["force_fidelity"]["per_channel_per_anatomical_region"]["bulk_reaction"]["7"]["status"], "fail")
        self.assertFalse(report["gates"]["measured_channel_fidelity_pass"])

    def test_green_strain_error_over_five_percent_fails(self):
        report = self.compare_logs(make_log(stretch=1.1), make_log(stretch=1.107))
        self.assertEqual(report["green_strain_fidelity"]["gate"], "fail")
        self.assertFalse(report["gates"]["measured_channel_fidelity_pass"])

    def test_zero_tag2_rows_are_not_applicable_not_missing(self):
        report = self.compare_logs(make_log(), make_log())
        force = report["force_fidelity"]
        self.assertEqual(force["tag2_attachment_force_applicability"],
                         "not_applicable_no_tag2_source_nodes")
        self.assertEqual(force["tag2_attachment_source_node_count"], 0)
        self.assertEqual(force["per_channel_per_object"]["tag2_attachment_force"]
                         ["Human/test-region"]["status"],
                         "not_applicable_no_tag2_source_nodes")
        self.assertEqual(force["gate"], "pass")

    def test_active_tag2_rows_fail_closed_when_unmapped(self):
        with self.assertRaises(module.ComparisonError):
            self.compare_logs(
                make_log(tag2_available="0", tag2_nodes=1),
                make_log(tag2_available="0", tag2_nodes=1))

    def test_active_tag2_force_is_compared(self):
        report = self.compare_logs(
            make_log(tag2_available="1", tag2_nodes=1, tag2_force=10.0),
            make_log(tag2_available="1", tag2_nodes=1, tag2_force=10.2))
        self.assertEqual(report["force_fidelity"]["tag2_attachment_force_applicability"],
                         "active_tag2_source_rows")
        self.assertEqual(report["force_fidelity"]["gate"], "fail")

    def test_unowned_tag2_force_is_rejected(self):
        with self.assertRaises(module.ComparisonError):
            self.compare_logs(
                make_log(tag2_available="1", tag2_nodes=0, tag2_force=10.0),
                make_log(tag2_available="1", tag2_nodes=0, tag2_force=10.0))

    def test_time_mismatch_is_rejected(self):
        with self.assertRaises(module.ComparisonError):
            self.compare_logs(make_log(dt="0.001"), make_log(dt="0.002"))

    def test_source_mass_mismatch_is_rejected(self):
        with self.assertRaises(module.ComparisonError):
            self.compare_logs(make_log(mass="1.0"), make_log(mass="1.01"))

    def test_zero_force_reference_is_not_counted_as_a_pass(self):
        report = self.compare_logs(make_log(reaction=0.0), make_log(reaction=0.0))
        self.assertEqual(report["force_fidelity"]["gate"], "incomplete_or_no_signal")
        self.assertFalse(report["gates"]["measured_channel_fidelity_pass"])

    def test_missing_force_channel_is_not_a_pass(self):
        report = self.compare_logs(make_log(contact_available="0"),
                                   make_log(contact_available="0"))
        self.assertNotEqual(report["force_fidelity"]["gate"], "pass")
        self.assertFalse(report["gates"]["measured_channel_fidelity_pass"])

    def test_reference_displacement_pair_defines_current_position(self):
        full = make_log(reference_mode=1)
        reduced = make_log(reference_mode=1)
        report = self.compare_logs(full, reduced)
        self.assertEqual(report["green_strain_fidelity"]["gate"], "pass")

    def test_per_tet_gate_catches_localized_error_hidden_by_volume_average(self):
        full = make_log(stretch=1.1, second_stretch=1.1, region_ids=(7, 9))
        reduced = make_log(stretch=1.1, second_stretch=1.107, region_ids=(7, 9))
        report = self.compare_logs(full, reduced)
        strain = report["green_strain_fidelity"]
        self.assertLess(strain["global_volume_weighted_relative_l1_error"], 0.05)
        self.assertEqual(strain["tetrahedra_error_at_or_over_threshold"], 1)
        self.assertEqual(strain["maximum_error_witness"]["anatomical_region_id"], 9)
        self.assertEqual(strain["per_anatomical_region"]["9"]["gate"], "fail")
        self.assertEqual(strain["gate"], "fail")

    def test_region_coverage_reports_registered_labels_and_unlabeled_gap(self):
        report = self.compare_logs(
            make_log(second_stretch=1.1, region_ids=(7, 9)),
            make_log(second_stretch=1.1, region_ids=(7, 9)))
        coverage = report["green_strain_fidelity"]["anatomical_region_coverage"]
        self.assertEqual(coverage["matched_labeled_tetrahedra"], 2)
        self.assertTrue(coverage["complete_for_all_source_tetrahedra"])
        gap_report = self.compare_logs(make_log(region_ids=()), make_log(region_ids=()))
        gap = gap_report["green_strain_fidelity"]["anatomical_region_coverage"]
        self.assertEqual(gap["matched_labeled_tetrahedra"], 0)
        self.assertFalse(gap["complete_for_all_source_tetrahedra"])
        self.assertIsNotNone(gap["coverage_gap"])

    def test_zero_reference_local_strain_error_is_counted_as_failure(self):
        report = self.compare_logs(
            make_log(stretch=1.0), make_log(stretch=1.000001))
        strain = report["green_strain_fidelity"]
        self.assertTrue(strain["maximum_relative_error_unbounded_due_to_zero_reference"])
        self.assertEqual(strain["tetrahedra_error_at_or_over_threshold"], 1)
        self.assertEqual(strain["gate"], "fail")

    def test_residual_is_kept_in_impulse_units(self):
        report = self.compare_logs(make_log(), make_log())
        self.assertEqual(report["mechanical_residual_impulse"]["units"], "N s")
        self.assertFalse(report["mechanical_residual_impulse"]["converted_to_newtons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
