from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from matter.tools.cardboard_opencourant_hex import (
    BlankConfig,
    StripConfig,
    build_fefco0201_blank,
    build_score_fold_strip,
    export_mesh_binary,
    export_mesh_json,
    _Profile,
    _frame_axes,
)


class CardboardOpenCourantHexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_score_fold_strip()

    def test_mesh_is_all_positive_hex8_with_source_recipe_geometry(self) -> None:
        mesh = self.mesh
        self.assertEqual(mesh.schema, "numi.cardboard.opencourant.hex-score-fold.v1")
        self.assertEqual(mesh.quality["hex8_count"], len(mesh.elements))
        self.assertGreater(mesh.quality["min_gauss_jacobian_mm3"], 0.0)
        self.assertEqual(mesh.quality["flute_centerline_peak_to_trough_mm"], 3.335)
        self.assertGreater(mesh.quality["material_volume_mm3"]["starch"], 0.0)
        self.assertEqual(set(mesh.quality["element_count_by_material"]),
                         {"liner", "medium", "starch"})
        for element in mesh.elements:
            self.assertEqual(len(element.nodes), 8)
            self.assertEqual(len(set(element.nodes)), 8)
            self.assertIn(element.material_role, {"liner", "medium", "starch"})
            self.assertEqual(len(element.frame_axes), 3)

    def test_flute_frame_metadata_follows_profile_tangent_and_is_right_handed(self) -> None:
        cfg = StripConfig().resolved()
        profile = _Profile(cfg)
        theta = profile.angle(cfg.pitch_mm / 8.0)
        md, cd, thickness = _frame_axes(theta)
        self.assertAlmostEqual(md[0], math.cos(theta))
        self.assertAlmostEqual(md[2], math.sin(theta))
        self.assertEqual(cd, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(thickness[0], -math.sin(theta))
        self.assertAlmostEqual(thickness[2], math.cos(theta))
        self.assertAlmostEqual(md[0] * cd[1] - md[1] * cd[0], thickness[2])
        self.assertAlmostEqual(md[1] * cd[2] - md[2] * cd[1], thickness[0])
        self.assertAlmostEqual(md[2] * cd[0] - md[0] * cd[2], thickness[1])
        self.assertGreater(md[2], 0.0, "positive-slope profile tangent must point toward +z")

    def test_finite_starch_and_independent_coincident_interface_faces(self) -> None:
        mesh = self.mesh
        self.assertEqual(set(mesh.quality["interface_pair_counts"]),
                         {"medium_starch", "starch_liner"})
        xyz = {node["id"]: tuple(node["xyz_mm"]) for node in mesh.nodes}
        for pair in mesh.interface_pairs:
            self.assertIsNone(pair["law_material_id"])
            a = pair["side_a"]["nodes"]
            b = pair["side_b"]["nodes"]
            self.assertEqual(len(a), 4)
            self.assertEqual(len(b), 4)
            self.assertTrue(set(a).isdisjoint(b))
            self.assertEqual(
                sorted(tuple(round(v, 8) for v in xyz[n]) for n in a),
                sorted(tuple(round(v, 8) for v in xyz[n]) for n in b),
            )
        kinds_by_patch: dict[int, set[str]] = {}
        for pair in mesh.interface_pairs:
            kinds_by_patch.setdefault(pair["patch_id"], set()).add(pair["interface"])
        self.assertEqual(len(kinds_by_patch), len(mesh.starch_patches))
        self.assertTrue(all(kinds == {"medium_starch", "starch_liner"}
                            for kinds in kinds_by_patch.values()))

    def test_score_tools_and_grips_are_boundary_sets(self) -> None:
        mesh = self.mesh
        self.assertEqual(mesh.score["state_at_export"],
                         "unscored; score must be physically formed by tool contact")
        self.assertGreater(len(mesh.surface_sets["score_punch_contact"]), 0)
        self.assertGreater(len(mesh.surface_sets["score_anvil_contact"]), 0)
        self.assertGreater(len(mesh.surface_sets["grip_left_contact"]), 0)
        self.assertGreater(len(mesh.surface_sets["grip_right_contact"]), 0)
        self.assertGreater(len(mesh.node_sets["grip_left"]), 0)
        self.assertGreater(len(mesh.node_sets["grip_right"]), 0)
        self.assertLess(mesh.tool_targets["indentation_mm"],
                        mesh.parameters["caliper_mm"])
        interface_refs = {
            (pair[side]["element_id"], pair[side]["face"])
            for pair in mesh.interface_pairs for side in ("side_a", "side_b")
        }
        free_refs = {
            (face.element_id, face.face)
            for face in mesh.surface_sets["all_free_surfaces"]
        }
        self.assertTrue(interface_refs.isdisjoint(free_refs))

    def test_stable_json_export_and_source_defaults(self) -> None:
        cfg = StripConfig().resolved()
        self.assertEqual((cfg.length_mm, cfg.width_mm, cfg.pitch_mm),
                         (31.6, 20.0, 7.9))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "strip.json"
            digest = export_mesh_json(self.mesh, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], self.mesh.schema)
            self.assertEqual(len(payload["elements"]), len(self.mesh.elements))
            self.assertEqual(len(digest), 64)

    def test_rsc_blank_has_real_flaps_tab_slots_and_score_tools(self) -> None:
        _, default_layout = BlankConfig().resolved_strip_config()
        self.assertEqual(default_layout["perimeter_span_mm"], 1025.0)
        self.assertEqual(default_layout["blank_height_mm"], 400.0)
        self.assertEqual(default_layout["internal_dimensions_mm"], None)
        self.assertEqual(default_layout["score_allowance_mm"], 0.0)
        self.assertEqual(default_layout["manufacturer_tab_width_mm"], 25.0)
        self.assertEqual(default_layout["slot_kerf_mm"], 1.0)
        config = BlankConfig(
            panel_length_mm=15.8,
            panel_width_mm=15.8,
            wall_height_mm=15.8,
            manufacturer_tab_width_mm=7.9,
            blank_axis_spacing_mm=5.0,
        )
        mesh = build_fefco0201_blank(config)
        self.assertEqual(mesh.quality["hex8_count"], len(mesh.elements))
        self.assertGreater(mesh.quality["min_gauss_jacobian_mm3"], 0.0)
        layout = mesh.score["layout"]
        self.assertEqual(layout["internal_dimensions_mm"], None)
        self.assertEqual(layout["score_allowance_mm"], 0.0)
        self.assertEqual(layout["manufacturer_tab_width_mm"], 7.9)
        self.assertEqual(layout["slot_kerf_mm"], 1.0)
        self.assertEqual(len(mesh.score["score_lines"]), 12)
        self.assertEqual(mesh.quality["blank_occupied_cell_area_mm2"],
                         2074.54)

        xyz = {node["id"]: tuple(node["xyz_mm"]) for node in mesh.nodes}
        panel_spans = layout["panel_spans_x_mm"]
        y_top, y_bottom = layout["flap_score_y_mm"]
        slot_centers = layout["slot_boundaries_x_mm"]
        half_kerf = 0.5 * layout["slot_kerf_mm"]
        material_cells = [
            element for element in mesh.elements
            if element.part in ("lower_liner", "upper_liner")
        ]
        found_top_flaps = [False] * len(panel_spans)
        found_bottom_flaps = [False] * len(panel_spans)
        found_tab_body = False
        for element in material_cells:
            points = [xyz[node_id] for node_id in element.nodes]
            cx = sum(point[0] for point in points) / 8.0
            cy = sum(point[1] for point in points) / 8.0
            if cy < y_top or cy > y_bottom:
                self.assertFalse(
                    any(center - half_kerf < cx < center + half_kerf
                        for center in slot_centers),
                    "slot strip contains a solid liner cell",
                )
                for i, (lo, hi) in enumerate(panel_spans):
                    if lo < cx < hi:
                        if cy < y_top:
                            found_top_flaps[i] = True
                        else:
                            found_bottom_flaps[i] = True
            elif layout["tab_span_x_mm"][0] < cx < layout["tab_span_x_mm"][1]:
                found_tab_body = True
        for element in mesh.elements:
            points = [xyz[node_id] for node_id in element.nodes]
            cx = sum(point[0] for point in points) / 8.0
            cy = sum(point[1] for point in points) / 8.0
            if cy < y_top or cy > y_bottom:
                self.assertFalse(
                    any(center - half_kerf < cx < center + half_kerf
                        for center in slot_centers),
                    f"slot strip contains a solid {element.material_role} cell",
                )
                self.assertFalse(
                    layout["tab_span_x_mm"][0] < cx < layout["tab_span_x_mm"][1],
                    "manufacturer tab incorrectly extends into a flap",
                )
        self.assertTrue(all(found_top_flaps))
        self.assertTrue(all(found_bottom_flaps))
        self.assertTrue(found_tab_body)
        self.assertEqual(set(mesh.quality["element_count_by_material"]),
                         {"liner", "medium", "starch"})

        cut_edges = {center: {center - half_kerf: 0, center + half_kerf: 0}
                     for center in slot_centers}
        for face in mesh.surface_sets["all_free_surfaces"]:
            if face.face not in ("x_minus", "x_plus"):
                continue
            points = [xyz[node_id] for node_id in face.nodes]
            cx = sum(point[0] for point in points) / 4.0
            cy = sum(point[1] for point in points) / 4.0
            if not (cy < y_top or cy > y_bottom):
                continue
            for boundaries in cut_edges.values():
                for edge in boundaries:
                    if abs(cx - edge) < 1.0e-8:
                        boundaries[edge] += 1
        self.assertTrue(all(count > 0 for row in cut_edges.values()
                            for count in row.values()))
        self.assertEqual(
            mesh.provenance["geometry_source_sha256"],
            "8db61e54ff4a0e2e4a47d6bcfe02244699b7820bf97474727bb77f1f9f389260",
        )

    def test_compact_binary_writer_avoids_expanded_json(self) -> None:
        mesh = build_fefco0201_blank(BlankConfig(
            panel_length_mm=15.8,
            panel_width_mm=15.8,
            wall_height_mm=15.8,
            manufacturer_tab_width_mm=7.9,
            blank_axis_spacing_mm=5.0,
        ))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "blank.ochex"
            digest = export_mesh_binary(mesh, target)
            payload = target.read_bytes()
        self.assertTrue(payload.startswith(b"OCHEX8\x00\x01"))
        header_length = int.from_bytes(payload[8:12], "little")
        header = json.loads(payload[12:12 + header_length])
        self.assertEqual(header["hex8_count"], len(mesh.elements))
        self.assertEqual(header["node_count"], len(mesh.nodes))
        self.assertEqual(len(digest), 64)
        self.assertLess(len(payload), 2_000_000)


if __name__ == "__main__":
    unittest.main()
