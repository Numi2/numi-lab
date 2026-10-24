from pathlib import Path
import sys,json,numpy as np,hashlib
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
from rebind_human_skin_reference import decode,rotation,registered_positions_float32
p=decode((E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin').read_bytes())
sp=json.loads((E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json').read_text());bp={r['body_index']:r for r in sp['all_bodies']}
w=p['weights'].astype(float);mats={}
for frame in ['rest','current']:
 A=[]
 for b in p['binds']:
  v=b['v'].astype(float);A.append(rotation(bp[int(b['body'])][frame]['orientation_world_xyzw'])@rotation(v[3:7])*v[7])
 mats[frame]=np.einsum('nk,kij->nij',w,np.array(A))
# C maps the locally blended rest linear coordinates into blended current linear coordinates.
C=np.linalg.solve(mats['rest'].transpose(0,2,1),mats['current'].transpose(0,2,1)).transpose(0,2,1)
sv=np.linalg.svd(C,compute_uv=False)
arr=np.load(E/'fine-source-wholehand-repair-001/wholehand-patch-arrays.npz')
dominant=np.array([int(x['body']) for x in p['binds']])[np.argmax(w,axis=1)]
def stats(ids):
 return {'vertices':len(ids),'minimum_singular_value_quantiles':np.quantile(sv[ids,-1],[0,.01,.05,.5,.95,1]).tolist(),'maximum_singular_value_quantiles':np.quantile(sv[ids,0],[0,.01,.5,.99,1]).tolist(),'determinant_quantiles':np.quantile(np.linalg.det(C[ids]),[0,.01,.5,.99,1]).tolist(),'vertices_min_singular_below_0_9':int((sv[ids,-1]<.9).sum()),'worst_vertex_ids':ids[np.argsort(sv[ids,-1])[:10]].tolist()}
r={'scope':'Frozen supported source-pose blend linear maps only; this is not accepted-state FEM deformation or a sweep over motion poses','sides':{}}
for side in ['left','right']:
 ids=arr[side+'_source_vertex_ids'].astype(int)
 r['sides'][side]={'all':stats(ids),'dominant_body_groups':{str(int(b)):stats(ids[dominant[ids]==b]) for b in np.unique(dominant[ids])}}
O=E/'regional-blend-conditioning-198';O.mkdir(exist_ok=True)
(O/'report.json').write_text(json.dumps(r,indent=2)+'\n')
for side in r['sides']:print(side,json.dumps(r['sides'][side]['all']))
