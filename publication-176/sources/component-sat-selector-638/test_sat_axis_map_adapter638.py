import numpy as np
import pytest

import sat_axis_map_adapter638 as adapter


def fixture():
    ids=np.array([[10,2,3,40],[10,2,3,41]],dtype=np.int64)
    nodes=np.array([[1,2,3],[1,2,3]],dtype=np.int64)
    inputs={
        "candidate":{"path":"/tmp/base.npz","sha256":"c"*64},
        "source_base":{"path":"/tmp/source.npz","sha256":"s"*64},
        "complete_159_report":{"path":"/tmp/audit159.json","sha256":"a"*64},
        "point_target_npz":{"path":"/tmp/point.npz","sha256":"p"*64},
        "face_target_npz":{"path":"/tmp/face.npz","sha256":"f"*64},
        "point_target_report":{"path":"/tmp/point.json","sha256":"r"*64},
        "face_target_report":{"path":"/tmp/face.json","sha256":"q"*64},
        "boundary_npz":{"path":"/tmp/boundary.npz","sha256":"g"*64},
        "boundary_report":{"path":"/tmp/boundary.json","sha256":"h"*64},
        "self_pair_seeds":{"path":"/tmp/seeds.npy","sha256":"i"*64},
    }
    report={"schema":"numi.connected_patch_sat_selector638.result.v1",
        "selected_pair_count":1,
        "preflight":{"schema":"numi.connected_patch_sat_selector638.preflight.v1",
            "pair_count":2,"inputs":dict(inputs,basis_scope_rows_sha256="scope",
                directional636_lower_sha256="lower",directional636_upper_sha256="upper")}}
    arrays={
        "pair_identity":ids,"pair_nodes":nodes,"selected":np.array([True,False]),
        "selected_normal_world_unit":np.array([[1.,0.,0.],[0.,0.,0.]]),
        "selected_axis_source":np.array(["skin_face_normal",""],dtype="U32"),
        "selected_axis_index":np.array([0,-1]),"selected_axis_sign":np.array([1,0]),
        "selected_axis_source_order":np.array([0,-1]),"selected_axis_sign_order":np.array([0,-1]),
        "selected_axis_extra_sign":np.array([1,0]),
        "displacement_node_rows":np.array([1,2,3],dtype=np.int64),
        "displacement_mm":np.zeros((3,3),dtype=np.float64),
    }
    return ids,nodes,inputs,report,arrays


def test_selector_inventory_and_shared_witness_validate():
    ids,nodes,inputs,report,arrays=fixture()
    result=adapter.validate_selector_arrays(arrays,report,ids,nodes,inputs,"scope","lower","upper")
    assert result["selected"].tolist()==[True,False]
    assert set(result["displacement_by_node"])=={1,2,3}


def test_one_skin_face_may_have_multiple_typed_bone_triangle_pairs():
    ids,nodes,inputs,report,arrays=fixture()
    assert np.array_equal(ids[0,:1],ids[1,:1])
    assert np.array_equal(nodes[0],nodes[1])
    result=adapter.validate_selector_arrays(arrays,report,ids,nodes,inputs,"scope","lower","upper")
    assert result["identity"].shape==(2,4)


def test_pair_callback_queues_are_reset_for_repeated_builder_calls():
    ids=np.array([[10,2,3,40],[10,2,3,41]],dtype=np.int64)
    nodes=np.array([[0,1,2],[0,1,2]],dtype=np.int64)
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bones={(2,3,40):np.array([[2.,0.,0.],[2.,1.,0.],[2.,0.,1.]]),
           (2,3,41):np.array([[3.,0.,0.],[3.,1.,0.],[3.,0.,1.]])}
    template=adapter.make_pair_queue_template(ids,nodes,points,bones)
    first=adapter.fresh_pair_queues(template)
    assert first[adapter._tri_key(nodes[0],points[nodes[0]],bones[(2,3,40)])].popleft()==0
    assert first[adapter._tri_key(nodes[1],points[nodes[1]],bones[(2,3,41)])].popleft()==1
    second=adapter.fresh_pair_queues(template)
    assert second[adapter._tri_key(nodes[0],points[nodes[0]],bones[(2,3,40)])].popleft()==0
    assert second[adapter._tri_key(nodes[1],points[nodes[1]],bones[(2,3,41)])].popleft()==1


def test_typed_identity_and_order_are_exact():
    ids,nodes,inputs,report,arrays=fixture()
    bad=ids[::-1].copy()
    with pytest.raises(ValueError,match="identity/order"):
        adapter.validate_selector_arrays(arrays,report,bad,nodes,inputs,"scope","lower","upper")


def test_runtime_box_hashes_must_match_exactly():
    ids,nodes,inputs,report,arrays=fixture()
    with pytest.raises(ValueError,match="lower-bound"):
        adapter.validate_selector_arrays(arrays,report,ids,nodes,inputs,"scope","stale","upper")


def test_selected_rows_need_all_three_shared_node_displacements():
    ids,nodes,inputs,report,arrays=fixture()
    arrays["displacement_node_rows"]=np.array([1,2],dtype=np.int64)
    arrays["displacement_mm"]=np.zeros((2,3))
    with pytest.raises(ValueError,match="lacks a displacement"):
        adapter.validate_selector_arrays(arrays,report,ids,nodes,inputs,"scope","lower","upper")


def test_selector_source_pins_are_required():
    pins={str(adapter.SAT_PATH):adapter.SAT_SHA,str(adapter.BOX_PATH):adapter.BOX_SHA,
          str(adapter.OWNER408_PATH):adapter.OWNER408_SHA,str(adapter.RUNNER458_PATH):adapter.RUNNER458_SHA,
          str(adapter.SELECTOR_HELPER_PATH):adapter.SELECTOR_HELPER_SHA,
          str(adapter.SELECTOR_RUNNER_PATH):adapter.SELECTOR_RUNNER_SHA}
    assert len(adapter.validate_selector_source_pins(pins))==6
    pins[str(adapter.SELECTOR_RUNNER_PATH)]="stale"
    with pytest.raises(ValueError,match="source pin mismatch"):
        adapter.validate_selector_source_pins(pins)


def test_axis_metadata_and_normal_must_match_551_inventory():
    axis={"axis_source":"skin_face_normal","axis_index":0,"axis_sign":1,
          "axis_source_order":0,"axis_sign_order":0,"extra_sign":1,
          "normal_world_unit":np.array([1.,0.,0.])}
    assert adapter.match_axis_record([axis],{"source":"skin_face_normal","index":0,
        "sign":1,"source_order":0,"sign_order":0,"extra_sign":1},[1.,0.,0.]) is axis
    with pytest.raises(ValueError,match="not unique"):
        adapter.match_axis_record([axis],{"source":"skin_face_normal","index":0,
            "sign":1,"source_order":0,"sign_order":0,"extra_sign":1},[0.,1.,0.])


def test_unselected_pair_uses_exact_637_fallback():
    fallback={"normal_world_unit":np.array([0.,1.,0.]),"corner_lower_mm":np.array([-1.,0.,0.])}
    result=adapter.route_pair_record(False,lambda:fallback,lambda:{"should_not":"run"})
    assert np.array_equal(result["normal_world_unit"],fallback["normal_world_unit"])
    assert result["axis_strategy"]=="finite_pair_sat_box_aware_soft_target_637_fallback"
    assert result["selector_pair_selected"] is False
    assert result["selector_displacement_applied_to_fit"] is False


def test_selected_pair_uses_verified_selector_record():
    record={"normal_world_unit":np.array([1.,0.,0.]),"axis_strategy":"joint_component_sat_box_witness_638"}
    result=adapter.route_pair_record(True,lambda:pytest.fail("fallback used"),lambda:record)
    assert result["axis_strategy"]=="joint_component_sat_box_witness_638"
    assert result["selector_pair_selected"] is True


def test_original_490_arguments_survive_nested_parser_removal():
    saved=["--base-candidate","base.npz","--source-base-npz","source.npz",
           "--audit159-490","audit159.json","--matched-inputs-json-490","matched.json"]
    env={"NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON":__import__("json").dumps(saved)}
    recovered=adapter.captured_fit_argv(env)
    assert adapter.arg_value(recovered,"--audit159-490").name=="audit159.json"
    assert adapter.arg_value(recovered,"--base-candidate").name=="base.npz"
    with pytest.raises(ValueError,match="did not preserve"):
        adapter.captured_fit_argv({})


def test_launcher_strips_only_its_two_explicit_flags():
    import run_axis_map_sat_fit638 as launcher
    argv=["runner.py","--base-candidate","b.npz","--selector-npz","sel.npz",
          "--selector-report","sel.json","--output","out"]
    kept,values=launcher.consume_flags(argv)
    assert kept==["runner.py","--base-candidate","b.npz","--output","out"]
    assert set(values)=={"--selector-npz","--selector-report"}
    with pytest.raises(ValueError,match="duplicate"):
        launcher.consume_flags(argv+["--selector-npz","second.npz"])



def test_real_697_selector_payload_matches_prepared_crossings():
    from pathlib import Path
    import json
    import sys

    d=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
    prep=d/"hard-face-feasibility-634/prepared-from697-001"
    out=d/"component-sat-selector-638/selector-697-002"
    current=prep/"current-inputs.json"
    report_path=out/"report.json"
    npz_path=out/"selection.npz"
    if not all(p.is_file() for p in (current,report_path,npz_path)):
        pytest.skip("frozen 697 selector output is not available on this checkout")
    runner_dir=d/"component-sat-selector-638"
    sys.path.insert(0,str(runner_dir))
    import run_connected_patch_sat_selector638 as runner
    import connected_patch_sat_selector638 as selector
    runner.selector=selector
    manifest=json.loads(current.read_text())
    prepared=prep/"prepared"
    runner.configure_preparation(prepared,current,Path(manifest["state"]))
    data=runner.prepare_current_inputs()
    report=json.loads(report_path.read_text())
    with np.load(npz_path,allow_pickle=False) as z:
        arrays={k:np.asarray(z[k]).copy() for k in z.files}
    expected={
        "candidate":{"path":data["candidate_path"],"sha256":data["candidate_sha256"]},
        "source_base":{"path":data["source_path"],"sha256":data["source_sha256"]},
        "complete_159_report":{"path":data["audit159_path"],"sha256":data["audit159_sha256"]},
        "point_target_npz":{"path":data["point_target_npz_path"],"sha256":data["point_target_npz_sha256"]},
        "face_target_npz":{"path":data["face_target_npz_path"],"sha256":data["face_target_npz_sha256"]},
        "point_target_report":{"path":data["point_target_report_path"],"sha256":data["point_target_report_sha256"]},
        "face_target_report":{"path":data["face_target_report_path"],"sha256":data["face_target_report_sha256"]},
        "boundary_npz":{"path":data["boundary_npz_path"],"sha256":data["boundary_npz_sha256"]},
        "boundary_report":{"path":data["boundary_report_path"],"sha256":data["boundary_report_sha256"]},
        "self_pair_seeds":{"path":data["self_pairs_path"],"sha256":data["self_pairs_sha256"]},
    }
    pre=report["preflight"]
    adapter.validate_selector_source_pins(pre["inputs"]["sources"])
    validated=adapter.validate_selector_arrays(arrays,report,data["pair_identity"],data["pair_nodes"],
        expected,data["scope_sha256"],data["lower_sha256"],data["upper_sha256"])
    assert len(validated["identity"])==697
    assert len(np.unique(validated["nodes"],axis=0))==160


def test_real_697_install_and_row_build_survive_490_argument_removal():
    from pathlib import Path
    import importlib
    import json
    import os
    import sys

    d=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
    prep=d/"hard-face-feasibility-634/prepared-from697-001"
    out=d/"component-sat-selector-638/selector-697-002"
    if not all(p.is_file() for p in (prep/"current-inputs.json",out/"report.json",out/"selection.npz")):
        pytest.skip("frozen 697 selected-axis result is not available")
    runner_dir=d/"component-sat-selector-638"
    sys.path.insert(0,str(runner_dir))
    import run_connected_patch_sat_selector638 as selector_runner
    import connected_patch_sat_selector638 as selector
    selector_runner.selector=selector
    manifest=json.loads((prep/"current-inputs.json").read_text())
    selector_runner.configure_preparation(prep/"prepared",prep/"current-inputs.json",Path(manifest["state"]))
    data=selector_runner.prepare_current_inputs()
    fit_argv_json=json.loads((prep/"prepared/fit-argv.json").read_text())
    fit_args=list(map(str,fit_argv_json[2:]))
    base=Path(manifest["candidate"]).resolve()
    audit=Path(manifest["surface_report"]).resolve()

    runner458=selector_runner.load_module("runner458_adapter638_smoke",
        selector_runner.RUNNER458_PATH,selector_runner.RUNNER458_SHA)
    owner=selector_runner.load_module("owner408_adapter638_smoke",
        selector_runner.OWNER408_PATH,selector_runner.OWNER408_SHA)
    helper_dir=d/"box-aware-sat-fit-637"
    sys.path.insert(0,str(helper_dir))
    helper637=importlib.import_module("box_aware_sat637")
    objective=helper637.load("objective458_adapter638_smoke",helper637.OBJ458,helper637.OBJ458_SHA)
    crossing=runner458.prepare_crossing_inputs(base,audit)

    old_argv=sys.argv[:]
    env_names=("NUMI_AXIS_MAP638_NPZ","NUMI_AXIS_MAP638_REPORT","NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON")
    old_env={k:os.environ.get(k) for k in env_names}
    try:
        sys.argv=["fit-smoke",*fit_args]
        os.environ["NUMI_AXIS_MAP638_NPZ"]=str(out/"selection.npz")
        os.environ["NUMI_AXIS_MAP638_REPORT"]=str(out/"report.json")
        os.environ["NUMI_AXIS_MAP638_ORIGINAL_FIT_ARGV_JSON"]=json.dumps(fit_args,separators=(",",":"))
        helper637.INSTALLS.clear()
        adapter.INSTALLS.clear()
        helper637.install_from_cli(owner,objective,crossing)
        for flag in ("--audit159-490","--matched-inputs-json-490"):
            i=sys.argv.index(flag)
            del sys.argv[i:i+2]
        state=adapter.install_axis_map_from_cli(owner,objective,crossing)
        # This calls only the in-memory 458 row builder; no LP, candidate write, audit, or geometry mutation.
        rows=objective.build_fixed_bone_outward_plane_rows(
            crossing["points_mm"],crossing["faces"],data["support_rows"],data["anchors"],
            crossing["pairs"],crossing["bone_triangles_mm"],total_weight=1.0)
        assert state["runtime_bounds_verified"] is True
        assert state["callbacks_consumed"]==697
        assert state["selected_pair_count"]==int(json.loads((out/"report.json").read_text())["selected_pair_count"])
        selected=helper637.INSTALLS[-1]["selected"]
        assert len(selected)==697
        assert sum(bool(x.get("selector_pair_selected")) for x in selected)==state["selected_pair_count"]
        assert len(rows["lower"])==rows["G"].shape[0]
    finally:
        sys.argv=old_argv
        for key,value in old_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        adapter.INSTALLS.clear()
        helper637.INSTALLS.clear()
