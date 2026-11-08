from pathlib import Path
import hashlib,json,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-quality-tolerances-20261108';B=R/'build/crow-quality-tolerances-build-01'
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def condition(H):
 scale=1/np.sqrt(np.diag(H));return float(np.linalg.cond(H*np.outer(scale,scale)))
plan=json.loads((A/'split-plan.json').read_text());out=B/'split-observations';out.mkdir();execution=[];started=time.monotonic()
for name,f in plan['files'].items():assert sha(R/name)==f['sha256']
for arm,f in plan['inputs'].items():
 assert sha(R/f['path'])==f['sha256']
 for repeat in range(2):
  assert time.monotonic()-started<plan['limits']['total_seconds']
  name=arm+f'-{repeat}';cmd=[B/'split-runtime/probe',B/'split-runtime/solver.metallib',R/f['path'],out/name];start=time.monotonic()
  with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
   try:code=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr,timeout=plan['limits']['per_command_seconds']).returncode
   except subprocess.TimeoutExpired:code=None
  rec=dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start);execution.append(rec);write(A/'split-execution.json',execution);print(json.dumps(rec),flush=True);assert code==0
rows=[];exact=True;prior={r['arm']:r for r in json.loads((A/'exploratory-analysis.json').read_text())['rows']}
for arm,f in plan['inputs'].items():
 c=ref.parse((R/f['path']).read_bytes());nv=c['nv'];p=out/(arm+'-0');status=struct.unpack('<16I16f',(p/'status.bin').read_bytes())
 values=np.fromfile(p/'scratch.f32x2',dtype='<f4').reshape(-1,2).astype(float).sum(axis=1);v=values[:nv]
 q=ref.evaluate(c,v);vref,qr,h=ref.reference(c);initial=ref.evaluate(c,c['start']);rounded=ref.evaluate(c,v.astype('f4').astype(float))
 same=all((out/(arm+'-0')/name).read_bytes()==(out/(arm+'-1')/name).read_bytes() for name in ['scratch.f32x2','status.bin']);exact &=same
 rows.append(dict(arm=arm,code=status[0],native_newton=status[1],native_backtracks=status[10],native_residual=status[16],independent_residual=q['res'],
 reference_velocity_abs_error=float(np.max(np.abs(v-vref))),rounded_residual=rounded['res'],declared_optimality=prior[arm]['declared_optimality'],
 rounded_satisfies_declared=rounded['res']<=prior[arm]['declared_optimality'],repeated_binary_exact=same,
 initial_equilibrated_hessian_condition=condition(initial['H']),final_equilibrated_hessian_condition=condition(q['H'])))
obs=dict(native_and_independent_residual_max=max(max(r['native_residual'],r['independent_residual']) if r['code']==0 else 1. for r in rows),
 velocity_reference_abs_error_max=max(r['reference_velocity_abs_error'] for r in rows),
 rounded_satisfies_original_declared_optimality=all(r['rounded_satisfies_declared'] and r['code']==0 for r in rows),repeated_binary_outputs_exact=exact)
supported={k:(v==plan['predictions'][k] if isinstance(v,bool) else v<=plan['predictions'][k]) for k,v in obs.items()}
write(A/'split-result.json',dict(schema='crow-quality-tolerances-retained-split-v1',rows=rows,observations=obs,predictions_supported=supported,all_predictions_supported=all(supported.values()),native_commands=len(execution),production_changed=False,goal_complete=False))
print((A/'split-result.json').read_text(),flush=True)
