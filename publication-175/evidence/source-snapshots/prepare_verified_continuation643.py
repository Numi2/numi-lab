#!/usr/bin/env python3
"""Build candidate-matched construction constraints; no physical admission.

All geometry/audits run via the existing Numi slot and 613 owner. The failed
candidate contributes self-fold witnesses; it is never admitted as geometry.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
N=ROOT/"D/surface-clearance-546"
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
def read(p):
 v=json.loads(Path(p).read_text())
 if not isinstance(v,dict):raise ValueError("expected JSON object: "+str(p))
 return v
def module(path):
 spec=importlib.util.spec_from_file_location("continuation546_643",path)
 if spec is None or spec.loader is None:raise ImportError(str(path))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for flag in ("state","prior-prepared","source-parent-state","failed-candidate","out"):
  ap.add_argument("--"+flag,type=Path,required=True)
 a=ap.parse_args();m=module(N/"continue_surface_round546.py")
 state=a.state.resolve();prior=a.prior_prepared.resolve();parent=a.source_parent_state.resolve();raw=a.failed_candidate.resolve();out=a.out.resolve()
 s=read(state);pr=read(prior/"prepared.json")
 if s.get("candidate_selected") is not False or s.get("native_admission") is not False or s.get("accepted_physical_steps")!=0:raise ValueError("state is not an unadmitted construction candidate")
 for k in ("candidate","point_summary","surface_report"):
  if m.sha(s[k])!=s.get(k+"_sha256"):raise ValueError("stale state role "+k)
 c=Path(s["candidate"]);p=Path(s["point_summary"]);surf=Path(s["surface_report"]);q=p.parent/"query/query-receipt.json"
 ps=read(p);bs=read(surf)
 if ps.get("candidate_sha256")!=m.sha(c) or ps.get("complete_surface_audit") is not True or ps.get("exact_PLC_surface_pairs")!=0 or ps.get("inside_node_bank_pairs")!=0:raise ValueError("current point/self baseline is not complete and zero")
 if bs.get("candidate_sha256")!=m.sha(c) or bs.get("complete") is not True or bs.get("bank_count")!=185 or bs.get("expected_bank_count")!=185:raise ValueError("current bank report incomplete")
 if not raw.is_file() or not q.is_file() or not parent.is_file():raise FileNotFoundError("missing failed proposal, query receipt, or parent state")
 out.mkdir(parents=True,exist_ok=False)
 audit=ROOT/"audit_connected_outer_geometry_130.py"
 m.run_stage([PYTHON,audit,raw,out/"failed-audit-130"],out/"failed-audit-stage",slot=True,source_paths=(Path(__file__),audit,raw),timeout=900)
 failed=read(out/"failed-audit-130/summary.json")
 if failed.get("candidate_sha256")!=m.sha(raw):raise ValueError("failed audit mismatches its candidate")
 owner=ROOT/"D/point-depth-barriers-357/point_depth_barriers_357.py";depth=ROOT/"D/point-depth-refresh-413/refresh_point_depth_rows.py"
 args=[PYTHON,depth,"--candidate",c,"--point-summary",p,"--surface-report",surf,"--query-dir",q.parent,"--output-dir",out/"depth-413","--expected-candidate-sha256",m.sha(c),"--expected-point-summary-sha256",m.sha(p),"--expected-surface-report-sha256",m.sha(surf),"--expected-query-receipt-sha256",m.sha(q),"--expected-owner-sha256",m.sha(owner)]
 m.run_stage(args,out/"depth-stage",slot=True,source_paths=(Path(__file__),state,c,p,surf,q,owner,depth),timeout=900)
 manifest={k:s[k] for k in ("candidate","candidate_sha256","point_summary","point_summary_sha256","surface_report","surface_report_sha256","native_admission","accepted_physical_steps")}
 manifest.update({"state":str(state),"state_sha256":m.sha(state),"depth_rows_npz":str(out/"depth-413/point-depth-rows.npz"),"depth_rows_report":str(out/"depth-413/365-compatible-report.json"),"query_receipt":str(q),"failed_candidate":str(raw),"failed_audit130":str(out/"failed-audit-130/summary.json"),"failed_pairs":str(out/"failed-audit-130/surface-self-pairs.npy"),"guard_context":str(prior/"constraints/constraint-context.json"),"prior_seed_report":str(prior/"constraints/self-seeds/seed-preflight.json"),"source_parent_context":str(parent)})
 m.write_json(out/"current-inputs.json",manifest)
 entry=N/"connected-continuation-613/prepare_matched613.py";fit=N/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
 args=[PYTHON,entry,"--current-inputs",out/"current-inputs.json","--fit-template",prior/"fit-argv.json","--matched-template",prior/"matched-inputs.json","--fit-entry",fit,"--prior-seed-report",prior/"constraints/self-seeds/seed-preflight.json","--out",out/"prepared"]
 m.run_stage(args,out/"prepare-stage",slot=False,source_paths=(Path(__file__),entry,fit,out/"current-inputs.json",state,parent,prior/"prepared.json"),timeout=2400)
 prep=out/"prepared";targets=prep/"surface-targets/targets.npz";z=np.load(targets,allow_pickle=False)
 expected={(int(f),int(b["body"]),int(b["stable_bone_id"]),int(t)) for b in bs["banks"] for f,t in ((b.get("result") or {}).get("triangle_pairs") or [])}
 actual=set(zip(*(map(int,z[k]) for k in ("face_rows","bank_body","bank_stable_bone_id","bank_triangle_stable_id"))))
 if len(expected)!=len(actual) or len(actual)!=len(z["face_rows"]) or expected!=actual or not np.all(z["target_exactly_outside_all185"]):raise ValueError("current typed target identities or endpoint checks mismatch")
 result={"status":"PASS_EXACT_CURRENT_PAIR_IDENTITIES_ONLY","current_pairs":len(expected),"bank_count":len(bs["banks"]),"all_sample_endpoints_outside_185":True,"audit_sha256":m.sha(surf),"target_npz_sha256":m.sha(targets),"prepared_sha256":m.sha(prep/"prepared.json"),"failed_self_witnesses":failed.get("exact_PLC_surface_pairs"),"source_script_sha256":m.sha(__file__),"native_admission":False,"accepted_physical_steps":0}
 m.write_json(prep/"independent-typed-identity-check.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
