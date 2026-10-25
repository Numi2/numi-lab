from pathlib import Path
import numpy as np,json,sys,hashlib,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/'source-connected-interior-007';P.mkdir(exist_ok=False)
sys.path.insert(0,str(D/'canonical/matter/tools'));import human_skin_fat_connected_repair as repair
I=Path('/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/retessellation-001/volume-cavity-001/oid-cavity-replacement.npz')
z=np.load(I);R=z['reference_points_AP_m'];O=z['target_points_AP_m'][:2411];T=z['candidate_tetrahedra'];B=z['candidate_O_faces'];mat=z['candidate_material_ids']
neighbors=[set()for _ in O]
for face in B:
 for i in face:neighbors[i].update(int(j)for j in face if j!=i)
Q=np.empty((2411,3,3));rank=[]
for i,nb in enumerate(neighbors):
 ids=sorted(nb);a=R[ids]-R[i];b=O[ids]-O[i]
 w=1/np.maximum(np.linalg.norm(a,axis=1),1e-12)**2
 u,s,v=np.linalg.svd((b*w[:,None]).T@a)
 Q[i]=u@np.diag([1,1,np.linalg.det(u@v)])@v
 rank.append(float(s[1]/max(s[0],1e-30)))
X=np.vstack([O]+[O+np.einsum('nij,nj->ni',Q,R[lo:lo+2411]-R[:2411])for lo in [2411,4822]]).astype(np.float32)
j,g,q,qq=repair.cell_values_gradients(X,R,T)
report={'initialization':'proper local neighbor polar transport of original O/I/D offsets; source O fixed; no runtime dynamics','input_sha256':hashlib.sha256(I.read_bytes()).hexdigest(),'initial_invalid_cells':int((j<=0).sum()),'initial_J_min':float(j.min()),'initial_J_max':float(j.max()),'initial_q_min':float(q.min()),'minimum_second_singular_ratio':min(rank),'rotation_orthogonality_max_error':float(np.max(np.abs(Q.transpose(0,2,1)@Q-np.eye(3)))),'geometry_admitted':False,'accepted_physical_steps':0}
np.savez_compressed(P/'initialization.npz',points_world_f32=X,rotations=Q,reference_points_m=R,tetrahedra=T,material_ids=mat)
start=time.monotonic()
Y,fit=repair.fit_connected_layer_interior(X,R,T,np.arange(2411,7233),max_iterations=48,linear_iterations=400)
report.update({'fit':fit,'elapsed_s':time.monotonic()-start,'solver_sha256':hashlib.sha256((D/'canonical/matter/tools/human_skin_fat_connected_repair.py').read_bytes()).hexdigest()})
np.savez_compressed(P/'candidate.npz',points_world_f32=Y,tetrahedra=T,material_ids=mat,reference_points_m=R)
report['candidate_sha256']=hashlib.sha256((P/'candidate.npz').read_bytes()).hexdigest()
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='fit'},indent=2));print(json.dumps(fit['final'],indent=2),flush=True)
