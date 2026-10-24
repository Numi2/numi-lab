from pathlib import Path
import importlib.util,json,hashlib,copy
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
O=E/'joint-surface-gradient-278'
O.mkdir(exist_ok=False)
helper=E/'batch-source-compose-252/tools/rebind_human_skin_reference.py'
s=importlib.util.spec_from_file_location('rb278',helper);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
source=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
basepose=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json'
b=r.decode(source.read_bytes());base=json.loads(basepose.read_text());F=b['inds']
poses={'current':basepose,'shoulder_elevation':E/'joint-source-decomposition-053/shoulder-elevation-050/registered-skin-poses.json','elbow_flexion':E/'joint-source-decomposition-053/elbow-flexion-080/registered-skin-poses.json','shoulder_plane':Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/shoulder_plane_bilateral_q35q73_m0p3_at_elevation0p5.json')}
manifest=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
names={int(x['core_body_index']):x['myosim_body'] for x in json.loads(manifest.read_text())['source']['anchors']}
dom=np.asarray([int(x['body']) for x in b['binds']])[np.argmax(b['weights'],axis=1)]
a=np.load(E/'fine-source-wholehand-repair-011/refinement-004/patch-arrays.npz')
adj=[set() for _ in range(b['nv'])]
for tri in F:
 for u,v in zip(tri,np.roll(tri,-1)):adj[int(u)].add(int(v));adj[int(v)].add(int(u))
regions={}
for side,suf in [('left','_l'),('right','_r')]:
 for region,tokens in [('shoulder',('clavicle','scapula')),('upper_arm',('humerus',)),('forearm',('radius','ulna'))]:
  ids=np.flatnonzero([names.get(int(x),'').endswith(suf) and any(t in names.get(int(x),'') for t in tokens) for x in dom])
  regions[side+'_'+region]=np.flatnonzero(np.any(np.isin(F,ids),axis=1))
 full={int(v) for v in a[side+'_source_vertex_ids'] if v<b['nv']};wrist=set(map(int,a[side+'_fixed_cut_ring_vertex_ids']))
 for _ in range(3):wrist|={v for u in wrist for v in adj[u] if v in full}
 for n,ids in [('hand',full),('wrist',wrist)]:regions[side+'_'+n]=np.flatnonzero(np.all(np.isin(F,list(ids)),axis=1))
X0=r.registered_positions_float32(b,base,frame='rest').astype(float)
E0=np.stack([X0[F[:,1]]-X0[F[:,0]],X0[F[:,2]]-X0[F[:,0]]],axis=2)
Q,R=np.linalg.qr(E0)
if np.any(np.abs(np.linalg.det(R))<1e-25):raise ValueError('singular source face')
Ri=np.linalg.inv(R)
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'source_sha256':sh(source),'helper_sha256':sh(helper),'source_pose_sha256':sh(basepose),'manifest_sha256':sh(manifest),'scope':'Diagnostic exact Float32 ABI5 source-surface differentials including skin-weight gradients; no FEM deformation or physical strain qualification. Regions based on dominant donor plus incident faces except hand/wrist fixed source ROI.','poses':{}}
for case,path in poses.items():
 p=json.loads(path.read_text())
 if 'source_poses' in p:
  p2=copy.deepcopy(base);by={int(x['core_body']):x for x in p['source_poses']}
  rows=p2.get('all_bodies',p2.get('bodies'))
  for row in rows:
   x=by.get(int(row['body_index']))
   if x is None:raise ValueError('missing source pose')
   row['current']={'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']}
  p=p2
 X=r.registered_positions_float32(b,p,frame='current').astype(float)
 E1=np.stack([X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]],axis=2)
 sv=np.linalg.svd(E1@Ri,compute_uv=False)
 rec={'pose_path':str(path),'pose_sha256':sh(path),'world_xyz_float32_sha256':hashlib.sha256(X.astype('<f4').tobytes()).hexdigest(),'regions':{}}
 for name,rows in regions.items():
  mi=sv[rows,1];ma=sv[rows,0];ix=rows[np.argsort(mi)[:12]]
  rec['regions'][name]={'faces':len(rows),'min_stretch_quantiles':np.quantile(mi,[0,.01,.05,.5,.95,1]).tolist(),'max_stretch_quantiles':np.quantile(ma,[0,.01,.05,.5,.95,1]).tolist(),'faces_below_0_5':int(np.sum(mi<.5)),'faces_above_2':int(np.sum(ma>2)),'worst_source_faces':ix.tolist(),'worst_source_vertices':F[ix].tolist()}
 out['poses'][case]=rec
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{n: {'min':v['min_stretch_quantiles'][0],'belowhalf':v['faces_below_0_5'],'max':v['max_stretch_quantiles'][-1]} for n,v in x['regions'].items()} for k,x in out['poses'].items()}))

