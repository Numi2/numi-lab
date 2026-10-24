from pathlib import Path
import csv,sys,runpy,json,hashlib
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');sys.path.insert(0,str(R/'tools'))
from rebind_human_skin_reference import decode,rotation
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
P=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003'
payload=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
p=decode(payload.read_bytes());poses=json.loads((P/'registered-skin-poses.json').read_text())
bp={int(b['body_index']):b for b in poses['all_bodies']}
m=runpy.run_path(str(E/'joint-region-roi-014/audit_exact_joint_region_intersections_015.py'));_,_,records,bv,bi,anchors=m['load_bones']()
rows=[52310,52297,51506]
def rot(q,p):
 q=np.asarray(q,dtype=np.float32);p=np.asarray(p,dtype=np.float32)
 x,y,z=np.moveaxis(p,-1,0);a,b,c,w=q
 u=2*(b*z-c*y);v=2*(c*x-a*z);t=2*(a*y-b*x)
 return np.stack([x+((w*u+b*t)-c*v),y+((w*v+c*u)-a*t),z+((w*t+a*v)-b*u)],axis=-1).astype(np.float32)
def winding(x,tri):
 a=tri[:,0]-x;b=tri[:,1]-x;c=tri[:,2]-x
 la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c))
 den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.arctan2(num,den).sum()/(2*np.pi))
out=[]
for vi in rows:
 worlds={};mats={};weights=p['weights'][vi]
 for phase in ['rest','current']:
  x=np.zeros(3,dtype=np.float32);A=np.zeros((3,3))
  for b,w in zip(p['binds'],weights):
   body=bp[int(b['body'])][phase];t=b['v']
   loc=(t[:3]+rot(t[3:7],p['verts'][vi,:3])*t[7]).astype(np.float32)
   v=(np.asarray(body['position_world_m'],np.float32)+rot(body['orientation_world_xyzw'],loc)).astype(np.float32)
   x=(x+w*v).astype(np.float32)
   A+=float(w)*(rotation(body['orientation_world_xyzw'])@rotation(t[3:7])*t[7])
  worlds[phase]=x.astype(float);mats[phase]=A
 checks=[]
 for r,anchor in zip(records,anchors):
  name=anchor['myosim_body']
  if name not in ['secondmc_l','3proxph_l','midph3_l']:continue
  body,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,_=r
  local=(np.array([tx,ty,tz],np.float32)+rot([qx,qy,qz,qw],bv[fv:fv+nv])*np.float32(scale)).astype(np.float32)
  F=bi[fi//3:(fi+ni)//3].astype(np.int64)-fv
  phases={}
  for phase in ['rest','current']:
   b=bp[body][phase];v=(np.asarray(b['position_world_m'],np.float32)+rot(b['orientation_world_xyzw'],local)).astype(np.float32).astype(float)
   phases[phase]=winding(worlds[phase],v[F])
  checks.append({'bone':name,'stable_id':stable,'winding':phases})
 Achange=mats['current']@np.linalg.inv(mats['rest'])
 out.append({'source_vertex':vi,'positions':{k:v.tolist()for k,v in worlds.items()},'weights':[{'body_index':int(b['body']),'weight':float(w)}for b,w in zip(p['binds'],weights)if w>1e-5],'current_from_rest_singular_values':np.linalg.svd(Achange,compute_uv=False).tolist(),'rest_lbs_det':float(np.linalg.det(mats['rest'])),'current_lbs_det':float(np.linalg.det(mats['current'])),'bone_checks':checks})
O=E/'source-vs-posed-hand-separation-192';O.mkdir(exist_ok=True)
report={'scope':'same pinned skin payload and bone sources; native-expression Float32 pose counterfactual, source rest is not accepted current state','witnesses':out,'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),'pose_sha256':hashlib.sha256((P/'registered-skin-poses.json').read_bytes()).hexdigest()}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
