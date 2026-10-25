from pathlib import Path
import numpy as np,json,sys,time,hashlib,shutil
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/'source-connected-interior-006';P.mkdir(exist_ok=False)
M=D/'canonical/matter/tools';sys.path.insert(0,str(M))
import human_skin_fat_connected_repair as repair
I=Path('/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/retessellation-001/volume-cavity-001/oid-cavity-replacement.npz')
assert hashlib.sha256(I.read_bytes()).hexdigest()=='0feda312f87a58ac8d7f73b9d27d870911d86040396ac0c65c24feeac682ee2b'
z=np.load(I);X=z['target_points_AP_m'].astype(np.float32);R=z['reference_points_AP_m'];T=z['candidate_tetrahedra'];mat=z['candidate_material_ids']
shutil.copy2(M/'human_skin_fat_connected_repair.py',P/'as-run-solver.py')
start=time.monotonic()
Y,report=repair.fit_connected_layer_interior(X,R,T,np.arange(2411,7233),max_iterations=48,linear_iterations=400)
report.update({'input_path':str(I),'input_sha256':hashlib.sha256(I.read_bytes()).hexdigest(),'solver_sha256':hashlib.sha256((P/'as-run-solver.py').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start})
np.savez_compressed(P/'candidate.npz',points_world_f32=Y,tetrahedra=T,material_ids=mat,reference_points_m=R)
report['candidate_sha256']=hashlib.sha256((P/'candidate.npz').read_bytes()).hexdigest()
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
