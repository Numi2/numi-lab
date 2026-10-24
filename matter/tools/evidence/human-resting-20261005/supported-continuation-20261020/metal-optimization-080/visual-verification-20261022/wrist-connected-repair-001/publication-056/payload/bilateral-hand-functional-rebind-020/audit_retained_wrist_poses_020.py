"""Exact source-replay regression at retained deterministic wrist poses.
Not a coupled simulation, accepted FEM state, ROM proof or native candidate capture.
"""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'bilateral-hand-functional-rebind-020'
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
import rebind_human_skin_reference as rb
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'run-001/retained-wrist-pose-regression';out.mkdir(exist_ok=False)
 source=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
 candidate=D/'run-001/candidate.nhskin'
 assert sha(source)=='d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04'
 assert sha(candidate)=='7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'
 a,b=[rb.decode(p.read_bytes()) for p in (source,candidate)]
 assert np.array_equal(a['inds'],b['inds']);faces=a['inds']
 results=[];all_pass=True
 report=dict(status='RUNNING',scope='Owner194 Float32 source replay at eight retained deterministic wrist probes. These are not breathing captures or candidate020 native runs.',
    source_sha256=sha(source),candidate_sha256=sha(candidate),
    audit_sha256=sha(__file__),exact_owner_sha256=sha(ci.__file__),replay_owner_sha256=sha(rb.__file__),
    cases=results,physical_admission=False,native_admission=False)
 for label in ['neutral','bilateral-flex','bilateral-extension','left-flex','left-extension','right-flex','right-extension','asymmetric-flex-extension']:
  started=time.monotonic()
  path=next((E/'native-pose-checks/repair-006'/label).glob('*.skin-poses.json'))
  snap=json.loads(path.read_text())
  pa,pb=[rb.registered_positions_float32(x,snap,frame='current') for x in (a,b)]
  ra,_=ci.float32_surface_records(pa,faces);rr,_=ci.float32_surface_records(pb,faces)
  cache,base=ci._prepare_complete_self_pair_baseline(ra)
  _,new=ci._update_complete_self_pair_baseline(cache,rr)
  oldpairs=set(map(tuple,base['triangle_pairs']));newpairs=set(map(tuple,new['triangle_pairs']))
  delta=newpairs-oldpairs
  area=lambda v: np.linalg.norm(np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]]).astype(float),axis=1)*.5
  aa,ab=area(pa),area(pb)
  row=dict(name=label,snapshot_path=str(path),snapshot_sha256=sha(path),baseline_exact_self_pairs=len(oldpairs),
   candidate_exact_self_pairs=len(newpairs),added_pair_count=len(delta),removed_pair_count=len(oldpairs-newpairs),
   added_pairs=[list(x) for x in sorted(delta)],candidate_degenerate_faces=np.flatnonzero(ab==0).tolist(),
   face_area_ratio_min=float(np.min(ab/aa)),face_area_ratio_max=float(np.max(ab/aa)),
   complete_surface_audit=new['complete_surface_audit'],elapsed_s=time.monotonic()-started)
  results.append(row);all_pass=all_pass and not delta and not np.any(ab==0)
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k not in ('added_pairs','candidate_degenerate_faces')}),flush=True)
  if not all_pass:break
 report['status']='PASS_NO_NEW_EXACT_SELF_PAIRS' if all_pass and len(results)==8 else 'REJECT_POSE_REGRESSION'
 report['completed_pose_count']=len(results)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report['status'],flush=True)
if __name__=='__main__':main()

