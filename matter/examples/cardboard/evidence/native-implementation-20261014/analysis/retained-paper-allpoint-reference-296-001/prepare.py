from pathlib import Path
import hashlib,json,sys,time
import numpy as np
b=Path('/Users/home/cardboard-implementation-20261014');out=b/'analysis/retained-paper-allpoint-reference-296-001';out.mkdir(exist_ok=False)
sys.path[:0]=[str(b/'analysis/bilinear15-fp64-replay-267'),str(b/'analysis/local-projection-scaled-correction-082')]
import bilinear15_reference as ref
import replay_scaled_policy as pol
capture=b/'solver/recycle-source-default-first-control-283/operator-capture';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t0=time.time()
header,rows=pol.material_rows(capture)
p=np.stack([r['p'] for r in rows]);f=np.stack([r['F'] for r in rows]);old=np.stack([r['old'] for r in rows]);direction=np.stack([r['direction'] for r in rows])
z,it,rnorm=ref.project(p,f,old,maxiter=80)
stress=ref.stress(p,f,z);tangent=ref.tangent(p,f,old,z,direction)
assert np.isfinite(z).all() and np.isfinite(stress).all() and np.isfinite(tangent).all()
np.savez_compressed(out/'reference.npz',keys=np.asarray([[r['environment'],r['tet_id'],r['material_index']] for r in rows],dtype=np.int32),parameters=p,F=f,accepted=old,direction=direction,projected=z,stress=stress,tangent=tangent,iterations=it,residual=rnorm)
r=dict(schema='numi.cardboard.retained-paper-allpoint-reference.v1',capture=str(capture),capture_sha256=h(capture),rows=len(rows),equations_sha256=h(b/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'),reader_sha256=h(b/'analysis/local-projection-scaled-correction-082/replay_scaled_policy.py'),artifact='reference.npz',artifact_sha256=h(out/'reference.npz'),elapsed_seconds=time.time()-t0,iterations_max=int(it.max()),scaled_residual_max=float(rnorm.max()),zero_direction_rows=int(np.count_nonzero(np.max(abs(direction),axis=(1,2))==0)),classification='Independent FP64 equations/complex-step implicit tangent, canonical FP32 operands plus captured low words. One captured direction perpoint. No Metal comparison, nearzero floor, FP64 finite-difference derivative gate, physical FEM trajectory, or convergence claim.')
(out/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(out/'prepare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(r),flush=True)
