from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
R=Path(__file__).parent
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module("reuse_test",R/"prepare_reused_coordinates.py")
def test_actual_current_report_accepts_changed_bank_counts():
 j=json.loads((R/"topology/check-003-filtered/audit-159/report.json").read_text())
 c=m.load(R/"topology/run-003-filtered/candidate.npz")
 seeds,scope=m.focus_scope(c["source_O_faces"],j)
 assert len(seeds)==1198 and len(scope)==1911
 j["banks"][1]=j["banks"][0]
 with pytest.raises(ValueError,match="duplicate"):m.focus_scope(c["source_O_faces"],j)
def test_actual_reused_candidate_preserves_outside_region_and_anchors():
 p=m.load(R/"topology/run-003-filtered/candidate.npz");c=m.load(R/"saved-reuse/run-004-focused/candidate.npz")
 j=json.loads((R/"topology/check-003-filtered/audit-159/report.json").read_text())
 _,scope=m.focus_scope(p["source_O_faces"],j);nodes=np.unique(p["source_O_faces"][scope]);outside=np.ones(len(p["points_world_m"]),bool);outside[nodes]=False
 assert np.array_equal(p["points_world_m"][outside].view(np.uint32),c["points_world_m"][outside].view(np.uint32))
 a=p["typed_coarse_anchor_output_rows_018"]
 assert np.array_equal(p["points_world_m"][a].view(np.uint32),c["points_world_m"][a].view(np.uint32))
 for key in p:
  if key!="points_world_m":
   assert p[key].dtype==c[key].dtype and p[key].shape==c[key].shape and p[key].tobytes()==c[key].tobytes()
def test_float32_edge_budget_matches_existing_context_owner(monkeypatch):
 for dirname,passes in [("run-002-focused",False),("run-003-focused",True)]:
  ctx=R/"saved-reuse"/dirname/"context.npz";g=module("grade_"+dirname,m.G)
  monkeypatch.setenv("NUMI_GRADED_LOCAL_STEP_CONTEXT_423",str(ctx));monkeypatch.setenv("NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423",m.sha(ctx));monkeypatch.setenv("NUMI_GRADED_LOCAL_STEP_KAPPA_423",".1")
  if passes:assert g.load_context_423()["kappa"]==.1
  else:
   with pytest.raises(ValueError,match="edge budgets"):g.load_context_423()
