from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P=Path(__file__).parent
B=P.parent/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
paths={'originalI':B/'emitted-OID-joint-nesting-309/I-closed-F32-stage.npz','derivedI':B/'minimum-thickness-minkowski-328/attempt-003/candidate-I-after-fat-f64.npz'}
results={}
for role,path in paths.items():
 t=time.perf_counter();z=np.load(path,allow_pickle=False)
 pk=next(x for x in ['points_world_m','points_world_m_f64','vertices','vert_properties'] if x in z.files)
 fk=next(x for x in ['faces','faces_f64','tri_verts'] if x in z.files)
 V=np.asarray(z[pk],np.float64)[:,:3];F=np.asarray(z[fk],np.int64)
 a=np.concatenate((F[:,0],F[:,1],F[:,2]));b=np.concatenate((F[:,1],F[:,2],F[:,0]))
 graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(V),len(V))).tocsr()
 count,label=connected_components(graph,directed=False)
 fl=label[F[:,0]]
 T=V[F];terms=np.einsum('ij,ij->i',T[:,0],np.cross(T[:,1],T[:,2]))/6
 components=[]
 for c in range(count):
  rr=np.flatnonzero(fl==c)
  if not len(rr):continue
  vertices=np.unique(F[rr]);q=V[vertices]
  components.append(dict(index=int(c),faces=int(len(rr)),vertices=int(len(vertices)),signed_volume_m3=float(np.sum(terms[rr])),bounds_min_m=q.min(0).tolist(),bounds_max_m=q.max(0).tolist(),vertex_mean_m=q.mean(0).tolist()))
 results[role]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),components=components,wall_s=time.perf_counter()-t)
receipt=dict(method='indexed_vertex_connected_components_and_oriented_surface_volume',results=results,physical_admission=False,clinical_or_anatomical_void_classification='not_inferred_from_sign_alone')
(P/'inner-shell-components.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
