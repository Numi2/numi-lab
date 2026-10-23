from pathlib import Path

import pytest

from matter.tools import cardboard_opencourant_hex as hex_mesh
from matter.tools import cardboard_opencourant_strip as strip


def test_rigidbody_uses_constituent_inertia_without_added_epsilon():
    lines = strip._rbody(7, 1007, 17, "left gripper").splitlines()
    data = lines[2].split()
    assert int(data[3]) == 2  # native Ispher=2 computes inertia from the body nodes
    assert float(data[4]) == 0.0  # no added mass
    assert [float(v) for v in lines[3].split()] == [0.0, 0.0, 0.0]
    assert [float(v) for v in lines[4].split()] == [0.0, 0.0, 0.0]


def test_type2_ties_use_documented_mass_conserving_spotflag_27():
    mesh = hex_mesh.build_score_fold_strip()
    role_by_part = {}
    for element in mesh.elements:
        role_by_part.setdefault(element.part, element.material_role)
    part_ids_by_role = {}
    for part_id, (part, role) in enumerate(sorted(role_by_part.items()), start=1):
        part_ids_by_role.setdefault(role, []).append(part_id)

    cards, _, metadata = strip._type2_ties(mesh, part_ids_by_role)
    assert len(cards) == 2
    assert {entry["spotflag"] for entry in metadata["interfaces"]} == {27}
    for card in cards:
        lines = card.splitlines()
        fields = lines[2].split()
        assert int(fields[3]) == 27
        # Pinned radioss2025 TYPE2 schema is
        # %20lg%20lg                    %10d (Istf at columns 61-70).
        fallback = lines[3].split()
        assert float(fallback[0]) == 1.0
        assert float(fallback[1]) == 0.05
        assert int(fallback[2]) == 2
        assert len(lines[3]) == 70
        assert float(lines[3][0:20]) == 1.0
        assert float(lines[3][20:40]) == 0.05
        assert lines[3][40:60] == " " * 20
        assert lines[3][60:70].strip() == "2"
    assert all(item["mass_addition"].endswith("no mass") for item in metadata["interfaces"])


def _node_group(deck: str, group_id: int) -> set[int]:
    lines = deck.splitlines()
    marker = f"/GRNOD/NODE/{group_id}"
    starts = [index for index, line in enumerate(lines) if line.strip() == marker]
    assert len(starts) == 1
    found = set()
    for line in lines[starts[0] + 2:]:
        if line.startswith("/"):
            break
        tokens = line.split()
        if tokens:
            found.update(int(token) for token in tokens)
    return found


def test_all_starch_nodes_remain_tied_and_rigid_grippers_drive_only_boundary_nodes():
    mesh = hex_mesh.build_score_fold_strip()
    _, parameters = strip.paper._read_parameters()
    part_names = sorted({element.part for element in mesh.elements})
    part_ids = {name: index for index, name in enumerate(part_names, start=1)}
    starter, _, _, case = strip._deck(
        mesh, parameters, part_ids, stem="paper-only-release", engine_mode="implicit-slow")

    roles_by_node = {}
    for element in mesh.elements:
        for node_id in element.nodes:
            roles_by_node.setdefault(int(node_id), set()).add(element.material_role)
    paper_grippers, _ = strip._paper_gripper_memberships(mesh)
    assert _node_group(starter, 5) == paper_grippers["left"]
    assert _node_group(starter, 6) == paper_grippers["right"]
    assert all(roles_by_node[node] <= {"liner", "medium"}
               for group_id in (5, 6) for node in _node_group(starter, group_id))

    interfaces = {item["interface"]: item
                  for item in case["interfaces"]["interfaces"]}
    assert set(interfaces) == {"medium_starch", "starch_liner"}
    for tie_id, interface in enumerate(sorted(interfaces), start=1):
        item = interfaces[interface]
        candidate_nodes = set()
        for pair in mesh.interface_pairs:
            if pair["interface"] != interface:
                continue
            secondary_side = "side_a" if pair["side_a_role"] == "starch" else "side_b"
            candidate_nodes.update(int(node) for node in pair[secondary_side]["nodes"])
        tied_nodes = _node_group(starter, 20 + tie_id)
        assert len(candidate_nodes) == 225
        assert tied_nodes == candidate_nodes
        assert not tied_nodes.intersection(_node_group(starter, 5) | _node_group(starter, 6))
        assert item["candidate_secondary_node_count"] == item["slave_node_count"] == 225
        assert item["secondary_nodes_omitted"] == item["excluded_gripper_slave_node_count"] == 0
        assert item["all_candidate_secondary_nodes_tied"] is True

    driven_groups = []
    lines = starter.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("/IMPDISP/", "/IMPVEL/")):
            row = lines[index + 2].split()
            driven_groups.append((line.split("/")[1], int(row[4])))
    assert driven_groups.count(("IMPDISP", 8)) == 3
    assert driven_groups.count(("IMPVEL", 4)) == 1
    assert len(driven_groups) == 4
    max_mesh_node = max(int(node["id"]) for node in mesh.nodes)
    assert _node_group(starter, 8) == {max_mesh_node + 2}  # right boundary master only
    assert _node_group(starter, 4) == {max_mesh_node + 3}  # analytic punch master only
    assert case["tools_and_boundary_conditions"]["gripper_membership"]["type2_rigidbody_main_compatibility"]["qualification"].startswith(
        "source/card semantics checked")


def test_type7_uses_all_free_paper_faces_and_dimensioned_gap():
    mesh = hex_mesh.build_score_fold_strip()
    group, surface, card, metadata = strip._type7_self_contact(mesh, end_time_s=12.0)
    element_roles = {int(element.id): element.material_role for element in mesh.elements}
    eligible = [face for face in mesh.surface_sets["all_free_surfaces"]
                if element_roles[int(face.element_id)] in {"liner", "medium"}]
    exact_ties = {
        (int(side["element_id"]), str(side["face"]))
        for pair in mesh.interface_pairs
        for side in (pair["side_a"], pair["side_b"])
    }
    assert metadata["main_surface"]["segment_count"] == len(eligible) - sum(
        (int(face.element_id), str(face.face)) in exact_ties for face in eligible)
    assert metadata["retained_adjacent_faces"]["sharing_gripper_nodes"] > 0
    assert metadata["gripper_slave_overlap_count"] > 0
    assert metadata["type2_secondary_overlap_count"] == 0
    assert "one-ring" not in metadata["excluded_faces"]["policy"]
    assert len(surface.splitlines()) == 2 + metadata["main_surface"]["segment_count"]
    assert group.startswith("/GRNOD/NODE/9\n")

    fields = card.splitlines()
    assert fields[0] == "/INTER/TYPE7/3"
    assert int(fields[2][40:50]) == 1000  # constant-gap mode
    assert float(fields[3][20:40]) == pytest.approx(strip.SELF_CONTACT_GAP_MM)
    assert int(fields[2][60:70]) == 0  # Ibag: no monitored volume is defined
    assert [float(value) for value in fields[5].split()] == [
        1.0, 0.0, strip.SELF_CONTACT_GAP_MM, 0.0, 12.0]
    assert fields[6][30:40].strip() == "1000"  # no initial coordinate adjustment


def test_strip_schedule_parks_tool_and_releases_boundary_grippers():
    mesh = hex_mesh.build_score_fold_strip()
    _, parameters = strip.paper._read_parameters()
    part_names = sorted({element.part for element in mesh.elements})
    part_ids = {name: index for index, name in enumerate(part_names, start=1)}
    starter, engine_1, engine_2, case = strip._deck(
        mesh, parameters, part_ids, stem="case11", engine_mode="implicit-slow")

    park = case["tools_and_boundary_conditions"]["score"]["park"]
    assert park["clearance_above_rigid_wing_envelope_mm"] == pytest.approx(2.0)
    assert park["end_s"] < case["tools_and_boundary_conditions"]["fold"]["start_s"]
    assert case["tools_and_boundary_conditions"]["fold"]["endpoint_dwell_s"] == 1.0
    assert "/SURF/SEG/33" in starter and "/INTER/TYPE7/3" in starter
    assert "/IMPL/DTINI\n0.1\n/IMPL/DT/1\n20 1.2 100 0.6666667" in engine_1
    assert engine_2.count("/RBODY/OFF") == 1
    assert "/IMPL/DYNA/2\n0.5 0.25" in engine_2
    assert case["engine_stages"][1]["mode"] == "implicit-dynamic"
    assert case["engine_stages"][1]["integration_controls"]["beta"] == 0.25
    assert "unrun and unqualified" in case["engine_stages"][1]["integration_controls"]["status"]
    assert f"{max(mesh.nodes, key=lambda node: node['id'])['id'] + 1:10d}" in engine_2
    assert case["engine_stages"][1]["released_rbody_ids"] == [1, 2]
    assert case["numerics"]["release_engine"]["explicit_timestep_forecast"][
        "solid_only_cycles_for_one_second"] == 97287594


def test_explicit_release_option_is_retained_separately():
    mesh = hex_mesh.build_score_fold_strip()
    _, parameters = strip.paper._read_parameters()
    part_names = sorted({element.part for element in mesh.elements})
    part_ids = {name: index for index, name in enumerate(part_names, start=1)}
    _, _, engine_2, case = strip._deck(
        mesh, parameters, part_ids, stem="explicit-option", engine_mode="explicit")
    assert "/DT/NODA/STOP" in engine_2
    assert "/IMPL/DYNA/2" not in engine_2
    assert case["engine_stages"][1]["mode"] == "explicit"
    assert "unexecuted option" in case["numerics"]["release_engine"]["status"]


def _write_mass_outputs(tmp_path: Path, *, added_mass: str, total_mass: str):
    starter = tmp_path / "starter.out"
    engine = tmp_path / "engine.out"
    starter.write_text(
        "TOTAL MASS AND MASS CENTER\n"
        "--------------------------\n"
        "MASS X Y Z\n"
        "3.9520258079227E-07 15.8 10.0 2.073\n"
        "TOTAL ADDED MASS = 0.000000000000\n"
        "TOTAL ADDED INERTIA = 0.000000000000\n",
        encoding="utf-8")
    engine.write_text(
        "CYCLE TIME TIME-STEP ELEMENT ERROR I-ENERGY K-ENERGY EXT-WORK "
        "MAS.ERR TOTAL MASS MASS ADDED\n"
        f"0 0.000 0.2500E-01 SOLID 1904 0.0% 0 0 0 0 0 0 "
        f"0.7751 {total_mass} {added_mass}\n",
        encoding="utf-8")
    return starter, engine


def test_mass_accounting_separates_tool_mass_and_detects_engine_added_mass(tmp_path):
    starter, engine = _write_mass_outputs(
        tmp_path, added_mass="0.3063E-06", total_mass="0.7015E-06")
    result = strip.collect_mass_accounting(
        starter, engine, known_rigid_tool_mass_Mg=0.0,
        tool_mass_in_starter=False,
        tool_mass_identity="massless analytic punch and supports")
    assert result["starter"]["board_mass_Mg"] == pytest.approx(3.9520258079227e-7)
    assert result["tooling"]["known_mass_Mg"] == 0.0
    assert result["tooling"]["external_to_solver_model_mass_Mg"] == 0.0
    assert result["engine_cycle_zero"]["added_mass_detected"] is True
    assert result["engine_cycle_zero"]["mass_balance_within_output_rounding"] is True
    assert result["mass_conservation_gate"]["passed"] is False


def test_mass_accounting_requires_known_tooling_identity_and_checks_zero_added_mass(tmp_path):
    starter, engine = _write_mass_outputs(
        tmp_path, added_mass="0.0", total_mass="0.3952E-06")
    result = strip.collect_mass_accounting(
        starter, engine, known_rigid_tool_mass_Mg=1.0e-8,
        tool_mass_in_starter=True, tool_mass_identity="known rigid tool")
    assert result["starter"]["board_mass_Mg"] == pytest.approx(3.8520258079227e-7)
    assert result["tooling"]["external_to_solver_model_mass_Mg"] == 0.0
    assert result["mass_conservation_gate"]["passed"] is True
    with pytest.raises(ValueError, match="tool_mass_identity"):
        strip.collect_mass_accounting(
            starter, engine, known_rigid_tool_mass_Mg=0.0,
            tool_mass_in_starter=False, tool_mass_identity=" ")
