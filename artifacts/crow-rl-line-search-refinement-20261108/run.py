from pathlib import Path
import hashlib,json,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01'
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());runtime=json.loads((A/'runtime.json').read_text())
for path,h in plan['sources'].items():assert sha(R/path)==h
for path,f in runtime['files'].items():assert sha(R/path)==f['sha256']
inputs=json.loads((A/'inputs.json').read_text());out=B/'observations'
if '--legacy-controls-only' not in sys.argv:out.mkdir()
execution=json.loads((A/'component-execution.json').read_text()) if '--legacy-controls-only' in sys.argv else []
for name,inp in inputs.items():
 assert sha(R/inp['path'])==inp['sha256']
 for arm,repeats,lib,legacy in [('baseline',2,B/'baseline.metallib','0'),('candidate',2,R/'build/shaders/MetalRobo.metallib','0'),('legacy',1,R/'build/shaders/MetalRobo.metallib','1'),('baseline-legacy',1,B/'baseline.metallib','1')]:
  if '--legacy-controls-only' in sys.argv and arm!='baseline-legacy':continue
  for repeat in range(repeats):
   label=f'{name}-{arm}-{repeat}';cmd=[B/'probe',lib,R/inp['path'],out/label,legacy];start=time.monotonic()
   with (out/(label+'.stdout')).open('xb') as stdout,(out/(label+'.stderr')).open('xb') as stderr:
    try:code=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr,timeout=30).returncode
    except subprocess.TimeoutExpired:code=None
   execution.append(dict(name=label,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start));write(A/'component-execution.json',execution);print(json.dumps(execution[-1]),flush=True);assert code==0
rows=[];same=True;preserved=True
for name,inp in inputs.items():
 raw=(R/inp['path']).read_bytes();d=struct.unpack_from('<16I8f',raw,32);nv,rs,nb=d[2:5];at=256+128*nb;blocks=raw[256:at]
 def take(n):
  global at
  x=np.frombuffer(raw,dtype='<f4',count=n,offset=at).copy();at+=4*n;return x
 M=take(nv*nv);J=take(rs*nv);bias=take(rs);free=take(nv);warm=take(nv)
 packed=struct.pack('<8I',0x53504331,1,nv,rs,nb,20,0,0)+blocks+b''.join(x.tobytes() for x in [M,J,bias,free,warm])
 c=ref.parse(packed);vref,qref,history=ref.reference(c)
 for arm in ['baseline','candidate','legacy']:
  p=out/(name+'-'+arm+'-0');status=struct.unpack('<12I12f',(p/'status.bin').read_bytes())
  v=np.fromfile(p/'velocity.f32',dtype='<f4')[:nv].astype(float);imp=np.fromfile(p/'impulse.f32',dtype='<f4')[:rs].astype(float)
  rec=dict(name=name,arm=arm,code=status[0],path=status[2],newton=status[4],backtracks=status[6],native_residual=status[12],native_feasibility=status[13])
  if status[0]==0:
   q=ref.evaluate(c,v);av=c['A']@(v-c['free']);jp=c['J'].T@imp;real=float(np.max(np.abs(av-jp))/(1+np.max(np.abs(av))+np.max(np.abs(jp))))
   rec.update(independent_projected_residual=q['res'],independent_published_residual=real,reference_velocity_abs_error=float(np.max(np.abs(v-vref))),original_optimality=d[16],original_feasibility=d[17],independent_feasibility=q['feas'])
   rec['independent_gate_passed']=q['res']<=d[16] and real<=d[16] and q['feas']<=d[17]
  else:rec['rejected_outputs_untouched']=all((p/f).read_bytes()==b'\xCD'*(4*max(nv,rs)) for f in ['velocity.f32','impulse.f32'])
  if arm!='legacy':
   exact=all((p/f).read_bytes()==(out/(name+'-'+arm+'-1')/f).read_bytes() for f in ['status.bin','velocity.f32','impulse.f32']);rec['repeat_exact']=exact;same &=exact
  rows.append(rec)
 base=out/(name+'-baseline-legacy-0');legacy=out/(name+'-legacy-0')
 preserved &=all((base/f).read_bytes()==(legacy/f).read_bytes() for f in ['status.bin','velocity.f32','impulse.f32'])
obs=dict(new_retained_candidate_cases_pass=sum(r['code']==0 and r.get('independent_gate_passed',False) for r in rows if r['arm']=='candidate' and r['name'] in ['api-default','opt6','opt7','line-search']),
 strict_substep7_candidate_rejected=all(r['code']==7 and r.get('rejected_outputs_untouched',False) for r in rows if r['arm']=='candidate' and r['name']=='substep7'),
 substep2_candidate_passed=all(r['code']==0 and r.get('independent_gate_passed',False) for r in rows if r['arm']=='candidate' and r['name']=='substep2'),
 legacy_arena_matches_baseline=preserved,repeated_binary_outputs_exact=same)
supported={k:v==plan['predictions'][k] for k,v in obs.items()}
write(A/'component-analysis.json',rows);write(A/'component-result.json',dict(observations=obs,predictions_supported=supported,all_predictions_supported=all(supported.values()),native_commands=len(execution),goal_complete=False))
print((A/'component-result.json').read_text(),flush=True)

assert all(supported.values()), 'Component predictions rejected; stop before physical qualification'
