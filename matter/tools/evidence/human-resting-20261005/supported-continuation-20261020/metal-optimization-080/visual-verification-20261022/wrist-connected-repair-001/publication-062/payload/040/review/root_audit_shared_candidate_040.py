"""Independent Float32 cell and boundary diagnostics for an unadmitted candidate."""
from pathlib import Path
import argparse,csv,json,hashlib,numpy as np
pa=argparse.ArgumentParser();pa.add_argument('--candidate',required=True);pa.add_argument('--output',required=True);a=pa.parse_args()
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
candidate=Path(a.candidate);z=np.load(candidate,allow_pickle=False);X=z['points_world_f32'].astype(np.float32).astype(float);T=z['tetrahedra'].astype(np.int64);mat=z['material_ids'].astype(int)
base=e/'native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv'
B=np.array([[float(r[c+'_m']) for c in 'xyz'] for r in csv.DictReader(base.open())],np.float32).astype(float)
reference=e/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv'
rr=list(csv.DictReader(reference.open()));Ref=np.empty_like(X)
for r in rr:Ref[int(r['local_volume_node'])]=[float(r[c+'_m']) for c in 'xyz']
def D(P):
 V=P[T];return np.stack([V[:,k]-V[:,0] for k in [1,2,3]],axis=2)
F=D(X)@np.linalg.inv(D(Ref));Fbase=D(B)@np.linalg.inv(D(Ref))
sv=np.linalg.svd(F,compute_uv=False);svb=np.linalg.svd(Fbase,compute_uv=False);J=np.linalg.det(F)
V=X[T];edge2=np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1)
det=np.linalg.det(D(X));q=det/np.maximum(edge2.max(axis=1),1e-30)**1.5
pp=e/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz';P=np.load(pp);assert np.array_equal(T,P['tetrahedra'])
Fouter=P['O_faces'].astype(int)
def surf(P):
 A=P[Fouter];N=np.cross(A[:,1]-A[:,0],A[:,2]-A[:,0]);area=np.linalg.norm(N,axis=1);return area,N/area[:,None]
ab,nb=surf(B);ac,nc=surf(X);rat=ac/ab;cos=np.einsum('ij,ij->i',nb,nc)
moved=np.flatnonzero(np.any(X!=B,axis=1));affected=np.flatnonzero(np.isin(T,moved).any(axis=1));fm=np.isin(Fouter,moved).any(axis=1)
def quant(v):return dict(zip(['min','p01','median','p99','max'],map(float,np.quantile(v,[0,.01,.5,.99,1]))))
layers={}
for layer,name in enumerate(['O','I','D']):
 ds=np.linalg.norm(X[layer*2411:(layer+1)*2411]-B[layer*2411:(layer+1)*2411],axis=1);layers[name]={'moved':int(np.count_nonzero(ds)),'max_displacement_m':float(ds.max())}
reports={}
for k,name in [(1,'skin'),(2,'fat')]:
 s=mat==k;reports[name]={'minimum_J':float(J[s].min()),'maximum_J':float(J[s].max()),'below_.35':int(np.sum(J[s]<.35)),'above_2.5':int(np.sum(J[s]>2.5)),'singular_values_all':quant(sv[s].ravel()),'baseline_singular_values_all':quant(svb[s].ravel()),'worst_cell_local_volume_row':int(np.flatnonzero(s)[np.argmax(sv[s,:,].max(axis=1))])}
r={'status':'INDEPENDENT_GEOMETRY_DIAGNOSTICS_NOT_NATIVE_ADMISSION','candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'moved_nodes':len(moved),'affected_cells':len(affected),'layers':layers,'materials':reports,'normalized_quality_minimum':float(q.min()),'nonpositive_cells':int(np.sum(det<=0)),'quality_below_1e_5':int(np.sum(q<1e-5)),'affected_outer_face_area_ratio':quant(rat[fm]) if fm.any() else None,'affected_outer_face_normal_cosine':quant(cos[fm]) if fm.any() else None,'outer_faces_flipped_vs_baseline':int(np.sum(cos<=0)),'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [candidate,base,reference,pp]},'limitations':['Singular values and area changes are diagnostic, not constitutive stress/energy admission.','No self-intersection, solid contact, detailed-skin mapping, mass reconstruction, retained motion, or accepted native step is implied.']}
out=Path(a.output);assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

