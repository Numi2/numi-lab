from pathlib import Path
import numpy as np,json,hashlib,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/'connected-layer-remesh-116/run-001'
def read(p):
 return [line.split('#')[0].split()for line in p.read_text().splitlines()if line.split('#')[0].strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
n=read(P/'whole-skin-fat-reference.1.node');e=read(P/'whole-skin-fat-reference.1.ele');f=read(P/'whole-skin-fat-reference.1.face')
assert len(n)==int(n[0][0])+1 and len(e)==int(e[0][0])+1
ids=np.asarray([int(r[0])for r in n[1:]],np.int64);R=np.asarray([[float(x)for x in r[1:4]]for r in n[1:]])
assert np.array_equal(ids,np.arange(1,len(R)+1))
T=np.asarray([[int(v)-1 for v in r[1:5]]for r in e[1:]],np.int64)
M=np.asarray([int(float(r[5]))for r in e[1:]],np.int8)
source=np.load(D/'connected-capacity-reference-115/candidate.npz')
assert np.array_equal(R[:7233],source['proposed_reference_points_m'][:7233])
X=R.astype(np.float32)
sys.path.insert(0,str(D/'connected-cell-trust-111/matter/tools'))
from human_skin_fat_connected_repair import cell_values_gradients
J,_,q,_=cell_values_gradients(X,R,T)
F=np.asarray([[int(v)-1 for v in r[1:4]]for r in f[1:]],np.int64);markers=np.asarray([int(r[4])for r in f[1:]],np.int64)
V=R[T];vol=np.einsum('ij,ij->i',V[:,1]-V[:,0],np.cross(V[:,2]-V[:,0],V[:,3]-V[:,0]))/6
np.savez_compressed(P/'candidate.npz',reference_points_m=R,points_world_f32=X,tetrahedra=T,material_ids=M,output_node_ids=ids,source_input_row=np.r_[np.arange(7233),np.full(len(R)-7233,-1)],surface_faces_zero_based=F,surface_face_markers=markers,cell_uids=np.asarray([int(r[0])for r in e[1:]],np.int64))
out={'points':len(R),'cells':len(T),'material_counts':{str(k):int((M==k).sum())for k in np.unique(M)},'generated_nodes':len(R)-7233,
 'positive_reference_cells':int((vol>0).sum()),'invalid_Float32_cells':int((J<=0).sum()),'below_quality_floor_1e_minus5':int((q<1e-5).sum()),
 'q_min':float(q.min()),'J_min_emitted_vs_new_reference':float(J.min()),'J_max_emitted_vs_new_reference':float(J.max()),
 'material_volume_m3':{str(k):float(vol[M==k].sum())for k in np.unique(M)},
 'candidate_sha256':sha(P/'candidate.npz'),'geometry_admitted':False,'accepted_physical_steps':0,
 'mass_mapping_regenerated':False,'detail_mapping_regenerated':False,'remaining_gates':'Full solid/bone/surrounding checks, generated ancestry, conservative mass/donor and detailed mapping, source-curve refinement and native qualification.',
 'construction_status':'Actual whole connected remesh of capacity-limited inner reference. Source O/I coordinates unchanged. New topology removes direct-reference-edit inversions; quality/clearance separately enforced.'}
(P/'cell-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
