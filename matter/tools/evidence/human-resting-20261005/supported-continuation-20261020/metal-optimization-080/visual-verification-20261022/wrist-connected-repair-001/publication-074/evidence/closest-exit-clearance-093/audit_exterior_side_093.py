from pathlib import Path
import numpy as np,json,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'closest-exit-clearance-093/run-001'
Z=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
z=np.load(Z);R=z['reference_points_m'];out=R[:2411]-R[4822:7233];out/=np.linalg.norm(out,axis=1)[:,None]
p=P/'iteration-001/contact-provenance.json';data=json.loads(p.read_text());res=[];excluded=0
for a in data:
 if a.get('body')!=23 or 'coefficients'not in a:continue
 coeff=a['coefficients'];ids=[c['node']for c in coeff]
 if max(ids)>=7233:excluded+=1;continue
 v=sum(c['weight']*out[c['node']%2411] for c in coeff);v/=np.linalg.norm(v)
 res.append({'cosine':float(np.asarray(a['nearest_exit_world_normal'])@v),'sample_query_row':a['sample_query_row'],'mechanics_node_rows':ids,'bone_stable_id':a['bone_stable_id']})
j={'method':'dot queried closest-exit covector with barycentric interpolation of normalized reference O-D column direction, original source nodes only','candidate_reference_sha256':hashlib.sha256(Z.read_bytes()).hexdigest(),'provenance_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'head_body':23,'eligible_edge_planes':len(res),'generated_support_planes_excluded':excluded,'opposed_planes':sum(x['cosine']<0 for x in res),'cosine_min':min(x['cosine']for x in res),'worst':sorted(res,key=lambda x:x['cosine'])[:30],'interpretation':'closest exit does not select the connected exterior side. This directional diagnostic alone is not an anatomical solid clearance certificate. Run093 is not an admitted repair.','physical_steps':0}
(P/'root-exterior-side-diagnostic.json').write_text(json.dumps(j,indent=2)+'\n')
