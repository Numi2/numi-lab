from pathlib import Path
import csv,json,hashlib,numpy as np
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
p=e/'bone-clearance-fit-041-exact-material-routes'
cp=e/'native-candidate020-all-bank-containment-035/source-preflight-003/all-bank-containment.csv'
bp=e/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.bodies.csv'
xp=e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
rows=[r for r in csv.DictReader(cp.open()) if r['classification']=='inside' and int(r['layer'])==0]
bodies={int(r['body_index']):r for r in csv.DictReader(bp.open())}
X=np.load(xp)['points'].astype(np.float32).astype(float)
cols=[];points=[];normals=[];body=[];bone=[];triangle=[];res=[]
for r in rows:
 i=int(r['local_node']);b=bodies[int(r['bank_body'])]
 M=np.array([float(b[f'world_to_body_m{a}{c}']) for a in range(3) for c in range(3)]).reshape(3,3)
 pos=np.array([b['position_'+c+'_m'] for c in 'xyz'],np.float32).astype(float)
 off=np.array([b['com_offset_'+c+'_m'] for c in 'xyz'],float)
 y=np.array([r['query_local_'+c] for c in 'xyz'],float)
 c=np.array([r['closest_'+c] for c in 'xyz'],float)
 err=float(np.linalg.norm(M@(X[i]-pos)-off-y));assert err<1e-12;res.append(err)
 nl=c-y;nl/=np.linalg.norm(nl);nw=M.T@nl;nw/=np.linalg.norm(nw)
 w=X[i]+np.linalg.solve(M,c-y);assert np.dot(nw,X[i]-w)<0
 cols.append(i);points.append(w);normals.append(nw);body.append(int(r['bank_body']));bone.append(int(r['bank_stable_id']));triangle.append(int(r['nearest_triangle']))
assert len(rows)==12
out=p/'baseline-outer-inside-local-planes.npz'
np.savez_compressed(out,constraint_local_node=np.array(cols,np.int64),constraint_closest_world_m=np.array(points),constraint_outward_normal_world=np.array(normals),constraint_body_index=np.array(body,np.int32),constraint_bone_stable_id=np.array(bone,np.int32),constraint_nearest_triangle=np.array(triangle,np.int64),constraint_margin_m=np.full(12,.0001))
r={'status':'12_BASELINE_OUTER_INSIDE_LOCAL_EXIT_CONSTRAINTS','count':12,'max_native_frame_residual_m':max(res),'npz_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [cp,bp,xp]},'limitations':['Nearest boundary planes for baseline O12 interior nodes only; not first-entry route witnesses or proof of network clearance.','All-bank node and segment queries mandatory after construction.']}
(p/'baseline-outer-inside-local-planes.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

