from pathlib import Path
import importlib.util,json,sys
import numpy as np
N=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('diag270',Path('/Users/numi/Work/numi-human-connected-surface-069/publication-109/sources/connected-refinement-270/connected_refinement_270.py'));o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
st=json.loads((N/'round-020-retained-sat552/final/state.json').read_text())
with np.load(st['candidate'],allow_pickle=False) as z:source={k:z[k] for k in z.files}
r=o._load_pinned(o.REFINER_PATH,o.REFINER_SHA256,'ref037diagnostic553')
X=source['points_world_m'];F=source['source_O_faces'];A=source['area_reference_037_points_world_m_f64']
rows,count=r.crossing_face_rows(json.loads(Path(st['surface_report']).read_text()),len(F))
result=r.p084_affine_refinement(X,F,rows,source['parent_source_face_marker'])
Q=o._csr_from_arrays(result,'source_prolongation_',len(result['points_world_m']),len(X))
Y=Q@A;G=result['source_O_faces'];parent=result['parent_face_rows']
cr=lambda p,f:np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
old=cr(A,F);new=cr(Y,G);dot=np.einsum('ij,ij->i',new,old[parent]);bad=np.flatnonzero(dot<=0)
w=[]
for k in bad:
 nodes=G[k];pr=int(parent[k]);qrows=[]
 for node in nodes:
  lo,hi=Q.indptr[node:node+2];qrows.append(dict(node=int(node),parents=Q.indices[lo:hi].tolist(),weights=Q.data[lo:hi].tolist()))
 w.append(dict(child=int(k),parent=pr,parent_nodes=F[pr].tolist(),child_nodes=nodes.tolist(),reference_dot=float(dot[k]),reference_parent_normal=old[pr].tolist(),reference_child_normal=new[k].tolist(),reference_cosine=float(dot[k]/(np.linalg.norm(old[pr])*np.linalg.norm(new[k]))),prolongation=qrows))
rp=Path(__file__).with_name('affine-winding-diagnosis.json');rp.write_text(json.dumps(dict(candidate_sha256=st['candidate_sha256'],old_nodes=len(X),new_nodes=len(Y),selected_crossing_faces=len(rows),bad_child_count=len(bad),bad_parent_rows=sorted({v['parent'] for v in w}),witnesses=w),indent=2)+'\n')
print(rp);print('badparents',sorted({v['parent'] for v in w}));print('witnesscount',len(w));print(w[:2])
