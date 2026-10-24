from pathlib import Path
import hashlib, json, struct, time, sys
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003'
CACHE=E/'registered-surrounding-tissue-native013-191'
C386=E/'source-route-frame-owner-386'
OUT=E/'source-route-frame-owner-387'; OUT.mkdir(parents=True,exist_ok=True)
BASE=Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
CAND=C386/'corrected-route-weights-candidate.nhtissue'
WANTED={131,132,133,134,137,138,141,142}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inv=json.loads((CACHE/'inventory.json').read_text())
assert inv['pose_path']==str(D/'registered-skin-poses.json')
poses={int(x['body_index']):x['current'] for x in json.loads((D/'registered-skin-poses.json').read_text())['all_bodies']}
def rotate(q,p):
 q=np.asarray(q,dtype=np.float32); p=np.asarray(p,dtype=np.float32)
 x,y,z=np.moveaxis(p,-1,0); a,b,c,w=q
 u=2*(b*z-c*y);v=2*(c*x-a*z);t=2*(a*y-b*x)
 return np.stack([x+((w*u+b*t)-c*v),y+((w*v+c*u)-a*t),z+((w*t+a*v)-b*u)],axis=-1).astype(np.float32)
def bodyworld(body,p):
 pose=poses[int(body)]
 return (np.asarray(pose['position_world_m'],dtype=np.float32)+rotate(pose['orientation_world_xyzw'],p)).astype(np.float32)
def parse_payload(path):
 raw=Path(path).read_bytes()
 magic,abi,nr,nb,nv,ni,reg,source=struct.unpack_from('<8s6I32s',raw)
 assert magic==b'NHTISS4\0' and abi==5
 off=64
 records=np.frombuffer(raw,dtype='<u4',count=nr*8,offset=off).reshape(-1,8);off+=nr*32
 bindtype=np.dtype([('body','<u4'),('transform','<f4',(8,))])
 bindings=np.frombuffer(raw,dtype=bindtype,count=nb,offset=off);off+=nb*36
 vtype=np.dtype([('p','<f4',(3,)),('n','<f4',(3,)),('binding','<u4',(4,)),('w','<f4',(4,))])
 verts=np.frombuffer(raw,dtype=vtype,count=nv,offset=off);off+=nv*56
 indices=np.frombuffer(raw,dtype='<u4',count=ni,offset=off);off+=ni*4
 assert off==len(raw)
 return raw,records,bindings,verts,indices
def emit_selected(path):
 raw,records,bindings,verts,indices=parse_payload(path)
 result={}
 for rec in records:
  fb,bc,fv,vc,fi,ic,stable,layer=map(int,rec)
  if stable not in WANTED: continue
  assert stable not in result, f'duplicate stable id {stable}'
  B=bindings[fb:fb+bc];v=verts[fv:fv+vc]
  assert len(B)==bc and len(v)==vc and fi+ic<=len(indices) and ic%3==0
  active=v['w']>2e-6
  assert np.all(np.abs(np.where(active,v['w'],0).sum(1)-1)<2e-3)
  assert np.all(v['binding'][active]<bc)
  used=np.unique(v['binding'][active]); bodyids=np.unique(B['body'][used])
  missing=[int(b) for b in bodyids if int(b) not in poses]
  if missing: raise ValueError(f'{stable}: missing current bodies {missing}')
  V=np.zeros((vc,3),dtype=np.float32)
  for slot in range(4):
   for local_binding in np.unique(v['binding'][active[:,slot],slot]):
    rows=np.flatnonzero(active[:,slot]&(v['binding'][:,slot]==local_binding))
    binding=B[int(local_binding)]; t=binding['transform']
    local=(t[:3]+rotate(t[3:7],v['p'][rows])*t[7]).astype(np.float32)
    world=bodyworld(binding['body'],local)
    V[rows]=(V[rows]+v['w'][rows,slot,None]*world).astype(np.float32)
  ix=indices[fi:fi+ic].astype(np.int64)
  if len(ix) and ix.max()<vc: F=ix.reshape(-1,3)
  elif len(ix) and ix.min()>=fv and ix.max()<fv+vc: F=(ix-fv).reshape(-1,3)
  else: raise ValueError(f'{stable}: invalid index range {ix.min()}..{ix.max()}')
  result[stable]=(V,F,dict(record=list(map(int,rec)),vertex_count=vc,face_count=len(F),body_ids=bodyids.astype(int).tolist()))
 if set(result)!=WANTED: raise ValueError(f'missing selected rows {sorted(WANTED-set(result))}')
 return result
base=emit_selected(BASE); cand=emit_selected(CAND)
cache=np.load(CACHE/'registered-tissue-banks.npz')
bank_by_stable={int(b['stable_id']):b for b in inv['banks'] if b['kind']=='soft_tissue' and int(b['stable_id']) in WANTED}
assert set(bank_by_stable)==WANTED
rows=[]; baseline_max=0.0; baseline_exact=True
for sid in sorted(WANTED):
 bv,bf,bmeta=base[sid]; cv,cf,cmeta=cand[sid]; bank=bank_by_stable[sid]; kv=cache[bank['key']+'_vertices']; kf=cache[bank['key']+'_faces']
 assert np.array_equal(bf,kf), f'{sid}: source/cache face order mismatch'
 assert np.array_equal(bv.shape,kv.shape)
 delta=np.abs(bv.astype(np.float64)-kv.astype(np.float64)); maxabs=float(delta.max()); maxnorm=float(np.linalg.norm((bv-kv).astype(np.float64),axis=1).max())
 baseline_max=max(baseline_max,maxnorm); baseline_exact &= (maxnorm < 1e-6 and np.array_equal(bf,kf))
 moved=np.linalg.norm((cv-bv).astype(np.float64),axis=1)*1000.0
 def q(a,p): return float(np.quantile(a,p))
 oldtri=bv[bf].astype(np.float64); newtri=cv[cf].astype(np.float64)
 oldn=np.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0]); newn=np.cross(newtri[:,1]-newtri[:,0],newtri[:,2]-newtri[:,0])
 olda=np.linalg.norm(oldn,axis=1); newa=np.linalg.norm(newn,axis=1)
 cos=np.einsum('ij,ij->i',oldn,newn)/np.maximum(olda*newa,1e-300)
 deg=int(np.count_nonzero(newa==0))
 area_ratio=newa/np.maximum(olda,1e-300)
 rows.append(dict(stable_id=sid,kind=bank['kind'],layer=bank['layer'],vertices=len(bv),faces=len(bf),baseline_vs_191_max_abs_axis_m=maxabs,baseline_vs_191_max_vertex_delta_m=maxnorm,baseline_faces_match_191=bool(np.array_equal(bf,kf)),candidate_displacement_mm=dict(p50=q(moved,.5),p95=q(moved,.95),max=float(moved.max())),candidate_face_degenerates=deg,candidate_vs_baseline_area_ratio=dict(min=float(area_ratio.min()),p01=q(area_ratio,.01),median=q(area_ratio,.5)),candidate_vs_baseline_normal_cosine=dict(min=float(cos.min()),negative=int(np.count_nonzero(cos<0))),candidate_vertices_sha256=hashlib.sha256(cv.tobytes()).hexdigest()))
report=dict(schema='numi.human.nhtiss.route-weight-counterfactual.current-pose-191.v1',candidate386_receipt_sha256=sha(C386/'candidate-receipt.json'),candidate_sha256=sha(CAND),baseline_payload_sha256=sha(BASE),identity003_path=str(D/'registered-skin-poses.json'),identity003_sha256=sha(D/'registered-skin-poses.json'),registered191_inventory_sha256=sha(CACHE/'inventory.json'),registered191_npz_sha256=sha(CACHE/'registered-tissue-banks.npz'),emission_code_path=str(E/'registered-surrounding-tissue-native013-191/prepare_registered_surrounding.py'),emission_code_sha256=sha(E/'registered-surrounding-tissue-native013-191/prepare_registered_surrounding.py'),selected_stable_ids=sorted(WANTED),pose='current from identity-003; identical rotate/bodyworld expression and payload lane order as prepare_registered_surrounding.py',baseline_match=dict(max_vertex_delta_m=baseline_max,all_faces_exact=all(r['baseline_faces_match_191'] for r in rows),per_bank_submicron=baseline_exact),rows=rows,gate_status='baseline_preflight_only' if baseline_exact else 'FAIL_INVALID_BASELINE_EMISSION',limitations=['This script verifies the current-pose Float32 CPU emission against the retained 191 cache and reports geometric displacement/face metrics only. No exact intersection, containment, soft-tissue or bone gate has run yet.'])
out=OUT/'current-pose-emission-preflight-001.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not baseline_exact: sys.exit(2)
