from pathlib import Path
import csv,hashlib,json,numpy as np
C=Path(__file__).parent;S=C.parents[3];B=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(p):
 with p.open() as f:return {(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])):float(r['nearest_distance_m'])*1000 for r in csv.DictReader(f) if r['classification']=='inside'}
def cross(p):
 j=json.loads(p.read_text());assert j['complete'] and j['bank_count']==185
 return {(int(b['body']),int(b['stable_bone_id']),int(f),int(t)) for b in j['banks'] for f,t in ((b.get('result') or {}).get('triangle_pairs') or [])}
x,y=inside(B/'audit-130/query/pairs.csv'),inside(C/'audit-130/query/pairs.csv');cx,cy=cross(B/'audit-159/report.json'),cross(C/'audit-159/report.json')
assert set(x)==set(y)
for name in [B,C]:
 j=json.loads((name/'audit-130/summary.json').read_text());assert j['complete_surface_audit'] and j['exact_PLC_surface_pairs']==0
with np.load(B/'candidate.npz') as z:a={k:z[k] for k in z.files}
with np.load(C/'candidate.npz') as z:
 assert set(a)==set(z.files)
 assert all(a[k].shape==z[k].shape and a[k].dtype==z[k].dtype and a[k].tobytes()==z[k].tobytes() for k in a if k!='points_world_m')
 before=a['points_world_m'].astype(float);after=z['points_world_m'].astype(float);faces=a['source_O_faces'];p=a['refinement_face_parent_face_row']
 changed=np.flatnonzero(np.any(before!=after,axis=1));f=np.flatnonzero(np.any(np.isin(faces,changed),axis=1))
 tr0=before[faces[f]];tr1=after[faces[f]];n0=np.cross(tr0[:,1]-tr0[:,0],tr0[:,2]-tr0[:,0]);n1=np.cross(tr1[:,1]-tr1[:,0],tr1[:,2]-tr1[:,0]);m0=np.linalg.norm(n0,axis=1);m1=np.linalg.norm(n1,axis=1);area=m1/m0;cos=(n0*n1).sum(axis=1)/(m0*m1)
parent0={(body,bone,int(p[face]),tri) for body,bone,face,tri in cx};parent1={(body,bone,int(p[face]),tri) for body,bone,face,tri in cy}
assert len(cx)<=985 and len(parent1)<=653 and min(cos)>.9 and area.min()>.75 and area.max()<1.3
assert max((y[k]-x[k] for k in x),default=0)<1e-5
banks=[(57,89),(59,95),(64,98),(114,117),(124,123)]
per=[]
for bank in banks:
 nodes=sorted(k for k in x if k[1:]==bank);changed_depth=[k for k in nodes if x[k]-y[k]>.001]
 assert changed_depth
 per.append({'bank':list(bank),'inside_nodes':len(nodes),'improved_nodes':len(changed_depth),'depth_before_sum_mm':sum(x[k] for k in nodes),'depth_after_sum_mm':sum(y[k] for k in nodes),'min_improvement_of_changed_mm':min(x[k]-y[k] for k in changed_depth)})
r={'status':'FULL_STATIC_DEPTH_IMPROVEMENT_ONLY','baseline_sha256':sha(B/'candidate.npz'),'candidate_sha256':sha(C/'candidate.npz'),'per_bank':per,'all_inside_pairs_before_after':[len(x),len(y)],'inside_added':0,'inside_removed':0,'all_bank_surface_pairs_before_after':[len(cx),len(cy)],'raw_pairs_added':[list(k) for k in sorted(cy-cx)],'raw_pairs_removed':[list(k) for k in sorted(cx-cy)],'parent_pairs_before_after':[len(parent0),len(parent1)],'parent_pairs_added':[list(k) for k in sorted(parent1-parent0)],'parent_pairs_removed':[list(k) for k in sorted(parent0-parent1)],'self_pairs':0,'changed_nodes':len(changed),'changed_faces':len(f),'maximum_displacement_mm':float(np.linalg.norm(after-before,axis=1).max()*1000),'minimum_face_area_ratio':float(area.min()),'maximum_face_area_ratio':float(area.max()),'minimum_face_normal_cosine':float(cos.min()),'all_nonposition_metadata_identical':True,'native_admission':False,'accepted_physical_steps':0,'limitations':['reference-pose construction only; surface pairs remain unresolved','inside-node samples do not establish solid clearance','skin-fat cell quality, mass, mechanical transfer, motion and native run remain untested']}
(C/'static-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
