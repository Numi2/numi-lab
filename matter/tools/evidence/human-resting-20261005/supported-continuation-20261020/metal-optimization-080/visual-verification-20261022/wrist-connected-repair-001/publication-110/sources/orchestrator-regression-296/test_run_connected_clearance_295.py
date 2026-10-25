
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/run_connected_clearance_295.py")
SPEC = importlib.util.spec_from_file_location("connected_clearance_295_tested", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_main_ast_routes_self_test_and_has_real_cli_bounds():
    tree=ast.parse(SOURCE.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    branch=next((n for n in ast.walk(main) if isinstance(n,ast.If)
                 and isinstance(n.test,ast.Attribute) and n.test.attr=="self_test"),None)
    assert branch is not None
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="self_test"
               for n in ast.walk(branch))
    assert any(isinstance(n,ast.Return) for n in ast.walk(branch))
    # The real parsed bounds are enforced before any pinned input or run access.
    assert "rounds must be in [1,3] and attempts-per-round in [1,2]" in SOURCE.read_text()

def test_cli_self_test_emits_json_and_its_builder_pin_assertion_is_live(monkeypatch,capsys):
    monkeypatch.setattr(sys,"argv",[str(SOURCE),"--self-test"])
    MOD.main()
    result=json.loads(capsys.readouterr().out)
    assert result["status"]=="PASS"
    assert "fixed prefix and area-reference keys" in result["checks"]
    assert "292 builder pin" in result["checks"]
    assert "empty pair seed shape" in result["checks"]
    # Prove this is an executed self-test with a real file-pin assertion,
    # rather than a JSON-shaped string or only constant comparisons.
    with monkeypatch.context() as m:
        m.setattr(MOD,"sha",lambda _path:"deliberately-wrong-pin")
        with pytest.raises(AssertionError):
            MOD.self_test()

@pytest.mark.parametrize("flag,value", [("--rounds","4"),("--attempts-per-round","3")])
def test_main_rejects_out_of_range_cli_limits_before_preflight(monkeypatch,flag,value,capsys):
    monkeypatch.setattr(sys,"argv",[str(SOURCE),flag,value,"--preflight-only"])
    with pytest.raises(SystemExit) as exc:
        MOD.main()
    assert exc.value.code==2
    assert "must be in [1,3]" in capsys.readouterr().err or "must be in [1,3]" in capsys.readouterr().out

def test_run_fit_attempt_emits_actual_n_by_3_delta_at_full_alpha_and_keeps_prefix(tmp_path,monkeypatch):
    out=tmp_path/"attempt";out.mkdir()
    candidate=tmp_path/"base.npz"
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float32)
    F=np.array([[0,2,3]],dtype=np.int64)
    anchors=np.array([0,1],dtype=np.int64)
    np.savez_compressed(candidate,points_world_m=X,source_O_faces=F,
                        typed_coarse_anchor_output_rows_018=anchors)
    solver=tmp_path/"solver.py";solver.write_text("# mock solver pin\n")
    monkeypatch.setattr(MOD,"SOLVER",solver)
    monkeypatch.setattr(MOD,"FIXED_PREFIX",2)
    monkeypatch.setattr(MOD,"require_headroom",lambda _stage:None)
    support=np.array([1,2,3],dtype=np.int64)
    delta=np.array([[.2,0.,0.],[2.,-3.,4.],[1.,2.,-1.]],dtype=np.float64)
    calls=[]
    def fake_run(command,log,*,slot=True,timeout=900):
        args=list(map(str,command)); proposal=Path(args[args.index("--output")+1])
        assert proposal==out/"proposal"
        assert not proposal.exists(), "fitter output must be created by the fitter"
        assert not Path(log).exists()
        calls.append({"slot":slot,"timeout":timeout,"proposal":proposal})
        proposal.mkdir()
        np.savez_compressed(proposal/"raw-lp-candidate.npz",
            nodes_lp_support_rows=support,nodes_lp_delta_mm_f64=delta)
        return {"returncode":0}
    monkeypatch.setattr(MOD,"run",fake_run)
    trial=MOD.run_fit_attempt({},candidate,np.empty((0,2),dtype=np.int64),
        tmp_path/"point.npz",tmp_path/"point.json",tmp_path/"face.npz",tmp_path/"face.json",out)
    assert trial==out/"trial-halfstep.npz" and trial.is_file()
    assert calls==[{"slot":True,"timeout":900,"proposal":out/"proposal"}]
    result=np.load(trial,allow_pickle=False)
    actual=np.asarray(result["points_world_m"])
    assert actual.dtype==np.float32
    assert np.array_equal(actual[:2].view(np.uint32),X[:2].view(np.uint32))
    # The fitter's (N,3) millimeter displacement is applied at alpha=1.
    expected=X.copy()
    expected[support]=((X[support].astype(np.float64)*1000.0+delta)*0.001).astype(np.float32)
    expected[anchors]=X[anchors]
    np.testing.assert_array_equal(actual,expected)
    receipt=json.loads((out/"trial-provenance.json").read_text())
    assert receipt["halfstep_alpha"]==1.0
    assert receipt["fixed_prefix_nodes"]==2 and receipt["fixed_prefix_bitwise_unchanged"] is True

def test_run_backtracker_creates_fresh_output_directory(tmp_path,monkeypatch):
    parent=tmp_path/"attempt";parent.mkdir()
    out=parent/"global-275";assert not out.exists()
    candidate=tmp_path/"candidate.npz";candidate.write_bytes(b"candidate fixture")
    trial=tmp_path/"trial.npz";trial.write_bytes(b"trial fixture")
    helper=tmp_path/"helper.py";helper.write_text("# helper fixture\n")
    backtracker=tmp_path/"backtracker.py";backtracker.write_text("# backtracker fixture\n")
    area=tmp_path/"area.npz";area.write_bytes(b"area fixture")
    monkeypatch.setattr(MOD,"BACKTRACKER",backtracker)
    monkeypatch.setattr(MOD,"AREA_REFERENCE",area)
    monkeypatch.setattr(MOD,"FIXED_PREFIX",0)
    seen={}
    def fake_run(command,log,*,slot=True,timeout=900):
        assert slot is False
        cfg=Path(list(map(str,command))[-1])
        assert cfg.is_file()
        seen.update(json.loads(cfg.read_text()))
        Path(log).write_text("mock completed without selected result\n")
    monkeypatch.setattr(MOD,"run",fake_run)
    selected=MOD.run_backtracker({"candidate":str(candidate),"parent_fit":"parent"},trial,out,
                                 helper,"reduce_global_step_scales",8,16)
    assert selected is None
    assert out.is_dir()
    assert seen["trial"]==str(trial)
    assert seen["fixed_prefix_nodes"]==0
    assert seen["maximum_exact_audits"]==8 and seen["maximum_construction_passes"]==16


# 298 directory-ownership regressions: uncached builders own output creation;
# only the validated cache-copy path creates the destination itself.
SOURCE298 = SOURCE.with_name("run_connected_clearance_298.py")
SPEC298 = importlib.util.spec_from_file_location("connected_clearance_298_tested", SOURCE298)
MOD298 = importlib.util.module_from_spec(SPEC298)
SPEC298.loader.exec_module(MOD298)

def _fixture_inputs(tmp_path):
    candidate = tmp_path / "candidate.npz"
    surface = tmp_path / "surface.json"
    candidate.write_bytes(b"candidate fixture bytes")
    surface.write_text('{"status":"PASS"}\\n')
    return candidate, surface

def test_298_uncached_face_builder_owns_fresh_output_directory(tmp_path, monkeypatch):
    candidate, surface = _fixture_inputs(tmp_path)
    out = tmp_path / "round"
    out.mkdir()
    target_dir = out / "face-targets"
    assert not target_dir.exists()
    builder = tmp_path / "builder.py"
    builder.write_text("# pinned mock builder\\n")
    monkeypatch.setattr(MOD298, "FACE_BUILDER", builder)
    seen = {}

    def fake_run(command, log, *, slot=True, timeout=900):
        args = list(map(str, command))
        assert args[0] == str(MOD298.PYTHON)
        assert args[1] == str(builder)
        destination = Path(args[args.index("--output-dir") + 1])
        assert destination == target_dir
        assert not destination.exists(), "uncached builder must receive a fresh path"
        assert args[args.index("--direction-mode") + 1] == "outward-nearest"
        assert slot is True
        destination.mkdir()
        np.savez_compressed(destination / "targets.npz", target_rows=np.array([3], dtype=np.int64))
        report = {
            "runner_sha256": MOD298.sha(builder),
            "direction_mode": "outward-nearest",
            "inputs": {"candidate_sha256": MOD298.sha(candidate)},
        }
        (destination / "report.json").write_text(json.dumps(report))
        seen["called"] = True

    monkeypatch.setattr(MOD298, "run", fake_run)
    npz, report = MOD298.run_face_targets(candidate, surface, out)
    assert seen["called"] is True
    assert npz == target_dir / "targets.npz" and npz.is_file()
    assert report == target_dir / "report.json" and report.is_file()

def test_298_cached_face_targets_create_copy_destination(tmp_path, monkeypatch):
    candidate, surface = _fixture_inputs(tmp_path)
    out = tmp_path / "round"
    out.mkdir()
    target_dir = out / "face-targets"
    cache = tmp_path / "validated-cache"
    cache.mkdir()
    builder = tmp_path / "builder.py"
    builder.write_text("# pinned mock builder\\n")
    monkeypatch.setattr(MOD298, "FACE_BUILDER", builder)
    monkeypatch.setattr(MOD298, "PINS", {
        **MOD298.PINS,
        "283_candidate_sha256": MOD298.sha(candidate),
        "283_surface_report_sha256": MOD298.sha(surface),
    })
    cached_npz = cache / "targets.npz"
    np.savez_compressed(cached_npz, target_rows=np.array([3, 4], dtype=np.int64))
    (cache / "target-records.jsonl").write_text('{"row":3}\\n')
    cached_report = cache / "report.json"
    cached_report.write_text(json.dumps({
        "runner_sha256": MOD298.sha(builder),
        "direction_mode": "outward-nearest",
        "inputs": {
            "candidate_sha256": MOD298.sha(candidate),
            "surface_report_sha256": MOD298.sha(surface),
        },
        "matched_159_complete": True,
        "passing_targets": 4002,
        "crossing_pairs": 4002,
        "failure_counts": {},
    }))
    monkeypatch.setattr(MOD298, "CACHED_293_REPORT_SHA256", MOD298.sha(cached_report))
    monkeypatch.setattr(MOD298, "CACHED_293_NPZ_SHA256", MOD298.sha(cached_npz))
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("validated cache path must not invoke the builder")
    monkeypatch.setattr(MOD298, "run", unexpected_run)

    assert not target_dir.exists()
    npz, report = MOD298.run_face_targets(candidate, surface, out, cache)
    assert target_dir.is_dir()
    assert npz.is_file() and report.is_file()
    assert MOD298.sha(npz) == MOD298.sha(cached_npz)
    assert MOD298.sha(report) == MOD298.sha(cached_report)
    assert (target_dir / "target-records.jsonl").read_text() == '{"row":3}\\n'
    cache_use = json.loads((target_dir / "cache-use.json").read_text())
    assert cache_use["cache_complete"] is True
    assert cache_use["targets_sha256"] == MOD298.sha(cached_npz)
