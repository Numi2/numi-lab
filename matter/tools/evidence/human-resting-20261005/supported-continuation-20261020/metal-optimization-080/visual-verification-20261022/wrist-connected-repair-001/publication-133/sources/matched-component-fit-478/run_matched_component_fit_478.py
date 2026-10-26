"""Explicitly matched461/476 component solver admission through pinned458/408 owners."""
from pathlib import Path
import hashlib,importlib.util,sys,json,os
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
W=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
H=D/"D/D/component-lp-decomposition-475/component_lp_decomposition_475.py"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 assert sha(W)=="08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
 assert sha(H)=="af0b6a6bf40757a403a8b7c359233694241d34e1e016b6dd3a37d7c98789c46d"
 w=load("matched458_for478",W);helper=load("component475_for478",H)
 w.EXPECTED_CANDIDATE_SHA="742315227c1d8225aac9307af828251a73a98a9c40b4f59fe791aeef5da1ba0f"
 w.EXPECTED_AUDIT_REPORT_SHA="5d4e7995c08245a197e5e12df688fafa95473dfd27b34aa12456085d79b589eb"
 w.EXPECTED_FACE_TARGET_SHA="e9c05ed4bb92fcd3197e43988b3d904df70bb81e58fa6880145e616d200e9694"
 w.EXPECTED_POINT_TARGET_SHA="f43022866e3dd64e947554dece63ca17ca7d19dbd34332b3df622ec606c3becf"
 assert sha(w.arg_value("--base-candidate"))==w.EXPECTED_CANDIDATE_SHA
 assert sha(w.arg_value("--area-reference-npz"))==w.EXPECTED_CANDIDATE_SHA
 assert os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458",w.EXPECTED_AUDIT_REPORT_SHA)==w.EXPECTED_AUDIT_REPORT_SHA
 original=w.install_objective
 def install(owner,objective,crossing):
  state=original(owner,objective,crossing);helper.install(owner,max_total_wall_s=120.0);return state
 w.install_objective=install
 w.main()
 out=w.arg_value("--output")/"report.json";report=json.loads(out.read_text());report["matched_component_fit_478"]=dict(pins={str(p):sha(p)for p in [W,H,Path(__file__)]},candidate_sha256=w.EXPECTED_CANDIDATE_SHA,point_target_sha256=w.EXPECTED_POINT_TARGET_SHA,face_target_sha256=w.EXPECTED_FACE_TARGET_SHA,physics_guards_unchanged=True,component_stage2_global_l1_optimality_claim=False,native_admission=False,accepted_physical_steps=0)
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
