from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-tolerances-20261108';B=R/'build/crow-quality-tolerances-build-01'
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'exploratory-plan.json').read_text());assert sha(A/'independent.py')==plan['independent_source_sha256']
calibration=ref.calibrate();rows=[];(B/'analysis').mkdir()
for arm,f in plan['inputs'].items():
 assert sha(R/f['path'])==f['sha256'];raw=(R/f['path']).read_bytes();stamp=struct.unpack_from('<8I',raw);d=struct.unpack_from('<16I8f',raw,32);status=struct.unpack_from('<12I12f',raw,128)
 nv,rs,nb=d[2:5];at=256+128*nb;blocks=raw[256:at]
 def take(n):
  global at
  x=np.frombuffer(raw,dtype='<f4',count=n,offset=at).copy();at+=4*n;return x
 M=take(nv*nv);J=take(rs*nv);bias=take(rs);free=take(nv);warm=take(nv);wp=take(rs);failed=take(nv)
 packed=struct.pack('<8I',0x53504331,1,nv,rs,nb,20,0,0)+blocks+b''.join(x.tobytes() for x in [M,J,bias,free,failed])
 (B/'analysis'/(arm+'-split-input.bin')).write_bytes(packed);c=ref.parse(packed)
 initial=ref.evaluate(c,failed.astype(float));v,q,history=ref.reference(c);rounded=ref.evaluate(c,v.astype('f4').astype(float))
 eigen=np.linalg.eigvalsh(initial['H']);eigen_final=np.linalg.eigvalsh(q['H'])
 count=lambda names:{k:names.count(k) for k in sorted(set(names))}
 for name,x in [('reference-v',v),('reference-p',q['p']),('reference-g',q['g'])]:(B/'analysis'/(arm+'-'+name+'.f64')).write_bytes(np.asarray(x,dtype='<f8').tobytes())
 rec=dict(arm=arm,capture_sha256=f['sha256'],substep=stamp[4],native_code=status[0],native_newton=status[4],native_pcg=status[5],native_backtracks=status[6],
 native_residual=status[12],independent_initial_residual=initial['res'],
 native_residual_relative_error=abs(initial['res']-status[12])/initial['res'],
 initial_projection_regions=count(initial['regions']),initial_hessian_eigen_min=float(eigen[0]),initial_hessian_condition=float(eigen[-1]/eigen[0]),
 fp64_reference_residual=q['res'],fp64_newton=len(history)-1,fp64_history=history,final_projection_regions=count(q['regions']),
 final_hessian_condition=float(eigen_final[-1]/eigen_final[0]),reference_velocity_difference_max=float(np.max(np.abs(v-failed))),
 rounded_reference_residual=rounded['res'],declared_optimality=d[16],rounded_reference_within_declared_optimality=rounded['res']<=d[16])
 rows.append(rec);print(json.dumps(rec),flush=True)
observed=dict(reference_reprojected_native_residual_relative_error_max=max(x['native_residual_relative_error'] for x in rows),fp64_reference_converges_80_steps=all(x['fp64_reference_residual']<=1e-13 and x['fp64_newton']<80 for x in rows),rounded_reference_satisfies_declared_optimality=all(x['rounded_reference_within_declared_optimality'] for x in rows))
write(A/'exploratory-analysis.json',dict(schema='crow-quality-tolerances-exploratory-v1',rows=rows,observations=observed,
 predictions_supported={k:(v==plan['predictions'][k] if isinstance(v,bool) else v<=plan['predictions'][k]) for k,v in observed.items()},calibration=calibration,production_changed=False,goal_complete=False))
