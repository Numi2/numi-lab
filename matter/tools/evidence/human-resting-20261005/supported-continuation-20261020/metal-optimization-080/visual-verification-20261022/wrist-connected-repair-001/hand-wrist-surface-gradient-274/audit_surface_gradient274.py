from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');O=E/'hand-wrist-surface-gradient-274';O.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('rb274',E/'batch-source-compose-252/tools/rebind_human_skin_reference.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
source=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin';snapshot=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json';b=r.decode(source.read_bytes());p=json.loads(snapshot.read_text());a=np.load(E/'fine-source-wholehand-repair-011/refinement-004/patch-arrays.npz')
X0=r.registered_positions_float32(b,p,frame='rest').astype(float);X1=r.registered_positions_float32(b,p,frame='current').astype(float);F=b['inds'];adj=[set() for _ in range(b['nv'])]
for tri in F:
 for u,v in zip(tri,np.roll(tri,-1)):adj[int(u)].add(int(v));adj[int(v)].add(int(u))
E0=np.stack([X0[F[:,1]]-X0[F[:,0]],X0[F[:,2]]-X0[F[:,0]]],axis=2);E1=np.stack([X1[F[:,1]]-X1[F[:,0]],X1[F[:,2]]-X1[F[:,0]]],axis=2)
Q,R=np.linalg.qr(E0);J=E1@np.linalg.inv(R);sv=np.linalg.svd(J,compute_uv=False);ratio=np.prod(sv,axis=1)
report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'regions':{},'scope':'Full Float32 source surface rest-to-current triangle differential including weight gradients, before accepted FEM transfer; diagnostic not physical strain or anatomy admission.'}
for side in ('left','right'):
 full={int(v) for v in a[side+'_source_vertex_ids'] if v<b['nv']};collar=set(map(int,a[side+'_fixed_cut_ring_vertex_ids']))
 for _ in range(3):collar|={v for u in collar for v in adj[u] if v in full}
 for name,ids in [('hand',full),('wrist',collar)]:
  rows=np.flatnonzero(np.all(np.isin(F,list(ids)),axis=1));order=rows[np.argsort(sv[rows,1])]
  report['regions'][side+'_'+name]={'faces':len(rows),'min_principal_stretch_quantiles':np.quantile(sv[rows,1],[0,.01,.05,.5,.95,1]).tolist(),'max_principal_stretch_quantiles':np.quantile(sv[rows,0],[0,.01,.05,.5,.95,1]).tolist(),'area_ratio_quantiles':np.quantile(ratio[rows],[0,.01,.05,.5,.95,1]).tolist(),'faces_below_0_5_stretch':int(np.sum(sv[rows,1]<.5)),'faces_above_2_stretch':int(np.sum(sv[rows,0]>2)),'worst_source_faces':order[:10].astype(int).tolist(),'worst_source_vertices':F[order[:10]].astype(int).tolist()}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

