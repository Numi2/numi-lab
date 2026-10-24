"""Retain exact remaining wrist-pose witnesses; no geometry edit."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'bilateral-hand-functional-rebind-020'
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci,classify
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
import rebind_human_skin_reference as rb
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'run-001/remaining-pose-witnesses';out.mkdir(exist_ok=False)
 asset=D/'run-001/candidate.nhskin';x=rb.decode(asset.read_bytes());faces=x['inds'];bodies=x['binds']['body']
 results=[]
 for label in ['right-flex','right-extension']:
  t=time.monotonic();p=next((E/'native-pose-checks/repair-006'/label).glob('*.skin-poses.json'));snap=json.loads(p.read_text())
  v=rb.registered_positions_float32(x,snap,frame='current');r,_=ci.float32_surface_records(v,faces)
  _,a=ci._prepare_complete_self_pair_baseline(r)
  rows=[]
  for ia,ib in a['triangle_pairs']:
   ia,ib=int(ia),int(ib);f1,f2=faces[ia],faces[ib]
   rows.append(dict(face_ids=[ia,ib],vertex_ids=[f1.tolist(),f2.tolist()],classification=classify(v[f1].astype(float),v[f2].astype(float)),
      coordinates_world_m=[v[f1].astype(float).tolist(),v[f2].astype(float).tolist()],
      strongest_body_ids=[bodies[x['weights'][f1].argmax(1)].tolist(),bodies[x['weights'][f2].argmax(1)].tolist()],
      shared_source_vertex_ids=np.intersect1d(f1,f2).tolist(),
      minimum_pair_vertex_distance_m=float(np.linalg.norm(v[f1,None,:].astype(float)-v[f2][None,:,:].astype(float),axis=2).min())))
  results.append(dict(name=label,snapshot_path=str(p),snapshot_sha256=sha(p),pairs=rows,elapsed_s=time.monotonic()-t))
 report=dict(status='UNRESOLVED_RETAINED_POSE_PAIR_WITNESSES',candidate_sha256=sha(asset),audit_sha256=sha(__file__),exact_owner_sha256=sha(ci.__file__),
  cases=results,physical_admission=False,native_admission=False,scope='Source-only exact pair localization. Interface classification is diagnostic; no automatic anatomical exemption.')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()

