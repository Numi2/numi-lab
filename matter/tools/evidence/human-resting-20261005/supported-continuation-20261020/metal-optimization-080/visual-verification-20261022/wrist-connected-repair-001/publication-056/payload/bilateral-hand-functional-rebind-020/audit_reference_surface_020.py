"""Exact same-pose detailed surface regression of hand influence recovery."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
import rebind_human_skin_reference as rb
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 begun=time.monotonic();out=D/'run-001/exact-reference-regression';out.mkdir(exist_ok=False)
 candidate=D/'run-001/candidate.nhskin';pos=D/'run-001/selected-hand-rows.npz'
 z=dict(np.load(pos));before=z['current_reference_positions_world_m'];after=z['rest_preserved_positions_world_m'];f=rb.decode(candidate.read_bytes())['inds']
 report=dict(status='RUNNING',scope='Same recorded source pose only; no motion or physical qualification',
   input_sha256={str(p):sha(p) for p in (candidate,pos,Path(__file__),Path(ci.__file__),Path(rb.__file__))},physical_admission=False,native_admission=False)
 try:
  br,_=ci.float32_surface_records(before,f);ar,_=ci.float32_surface_records(after,f)
  cache,baseline=ci._prepare_complete_self_pair_baseline(br)
  _,result=ci._update_complete_self_pair_baseline(cache,ar)
  a=set(map(tuple,result['triangle_pairs']));b=set(map(tuple,baseline['triangle_pairs']))
  report.update(status='PASS_NO_NEW_EXACT_SELF_PAIRS' if not a-b else 'REJECT_NEW_EXACT_SELF_PAIRS',
    baseline_self_pairs=len(b),candidate_self_pairs=len(a),added_pairs=[list(p) for p in sorted(a-b)],
    removed_pairs=[list(p) for p in sorted(b-a)],baseline_aabb_candidates=baseline['aabb_candidate_pairs'],
    candidate_reaudited_faces=result['selected_face_count'],complete_surface_audit=result['complete_surface_audit'])
 except Exception as exc:
  report.update(status='EXACT_GATE_ERROR',error=str(exc))
 report['elapsed_s']=time.monotonic()-begun
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('added_pairs','removed_pairs')}),flush=True)
if __name__=='__main__':main()
