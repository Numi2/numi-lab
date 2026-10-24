from pathlib import Path
import sys,json,runpy,hashlib
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');sys.path.insert(0,str(R/'tools'))
from rebind_human_skin_reference import decode,rotation,positions
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
skinpath=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
posepath=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json'
p=decode(skinpath.read_bytes());js=json.loads(posepath.read_text());bodies={b['body_index']:b['rest']for b in js['all_bodies']}
AA=[];bb=[]
for b in p['binds']:
 t=b['v'].astype(float);bp=bodies[int(b['body'])];rr=rotation(bp['orientation_world_xyzw'])
 AA.append(rr@rotation(t[3:7])*t[7]);bb.append(np.asarray(bp['position_world_m'])+rr@t[:3])
X=positions(p,np.array(AA),np.array(bb));F=p['inds'].astype(np.int64)
tri=X[F];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);vn=np.zeros_like(X)
for k in range(3):np.add.at(vn,F[:,k],fn)
norm=np.linalg.norm(vn,axis=1);sn=p['verts'][:,3:6].astype(float);snorm=np.linalg.norm(sn,axis=1)
valid=(norm>1e-16)&(snorm>1e-12);cos=np.zeros(len(X));cos[valid]=np.einsum('ij,ij->i',vn[valid],sn[valid])/(norm[valid]*snorm[valid])
m=runpy.run_path(str(E/'joint-region-roi-014/audit_exact_joint_region_intersections_015.py'));manifest=json.loads(m['MAN'].read_text());names={int(a['core_body_index']):a['myosim_body']for a in manifest['source']['anchors']}
donor=p['binds']['body'][np.argmax(p['weights'],axis=1)]
def stats(ii):
 c=cos[ii];return{'vertices':len(c),'negative':int(np.sum(c<0)),'below_05':int(np.sum(c<.5)),'median_cosine':float(np.median(c)),'p01_cosine':float(np.quantile(c,.01)),'worst_source_vertex_ids':ii[np.argsort(c)[:10]].tolist()}
groups={names.get(int(i),str(i)):stats(np.flatnonzero(valid&(donor==i)))for i in np.unique(donor)if np.any(valid&(donor==i))}
j={'scope':'Numerical reference-normal diagnostic; smoothed authored normals are not expected to equal geometric normals exactly. No source edit or visual qualification.','skin_sha256':hashlib.sha256(skinpath.read_bytes()).hexdigest(),'pose_sha256':hashlib.sha256(posepath.read_bytes()).hexdigest(),'whole':stats(np.flatnonzero(valid)),'unreferenced_or_zero_geometry_normals':int(np.sum(norm<=1e-16)),'by_dominant_body':groups}
O=E/'native-rest-skin-normal-audit-193';O.mkdir(exist_ok=True);(O/'report.json').write_text(json.dumps(j,indent=2)+'\n')
print(json.dumps({'whole':j['whole'],'worst_bodies':sorted(groups.items(),key=lambda x:-x[1]['negative'])[:12]}))
