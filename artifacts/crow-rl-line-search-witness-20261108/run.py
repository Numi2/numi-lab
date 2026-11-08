from pathlib import Path
import hashlib,json,os,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-witness-20261108';B=R/'build/crow-line-search-witness-build-01';P=B/'reference';O=B/'observations';O.mkdir()
sys.path.insert(0,str(A));import independent as ref
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());seal=json.loads((A/'runtime.json').read_text())
for p,h in plan['sources'].items():assert sha(R/p)==h
for p,h in seal['files'].items():assert sha(R/p)==h
env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(B/'runtime/metalrobo/MetalRobo.metallib')}
jobs=[('quality-owner',[B/'runtime/quality-owner'],True),('contact-owner',[B/'runtime/contact-owner',B/'runtime/metalrobo/MetalRobo.metallib'],True),
 ('api-0',[P/'crow-fixture',O/'api-0','0','1','32','2','api-default'],False),
 ('api-1',[P/'crow-fixture',O/'api-1','0','1','32','2','api-default'],False),
 ('strict',[P/'crow-fixture',O/'strict','0','1','32','2','strict'],False)]
execution=[]
for name,cmd,success in jobs:
 local=dict(env)
 if name.startswith('api') or name=='strict':local['METALROBO_QUALITY_FAILURE_CAPTURE']=str(O/(name+'-capture.bin'))
 start=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as stdout,(O/(name+'.stderr')).open('xb') as stderr:
  code=subprocess.run([str(x) for x in cmd],env=local,stdout=stdout,stderr=stderr,timeout=180).returncode
 execution.append(dict(name=name,returncode=code,seconds=time.monotonic()-start));write(A/'execution.json',execution);print(json.dumps(execution[-1]),flush=True)
 assert code in ([0] if success else [1])
 assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib')) for line in (O/(name+'.stderr')).read_text().splitlines())
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<plan['limits']['raw_bytes']
matched={}
for name in ['api-0','api-1']:
 matched[name]=[p.name for p in (P/'api-gold').iterdir() if p.read_bytes()==(O/name/p.name).read_bytes()]
 assert len(matched[name])==len(list((P/'api-gold').iterdir()))
gold=(P/'strict-gold.bin').read_bytes();strict=(O/'strict-capture.bin').read_bytes();assert gold==strict
raw=(O/'api-0-capture.bin').read_bytes();assert raw==(O/'api-1-capture.bin').read_bytes()
h=struct.unpack_from('<8I',raw);d=struct.unpack_from('<16I8f',raw,32);status=struct.unpack_from('<12I12f',raw,128);nv,rows,nb=d[2:5]
assert h[:2]==(0x51464c31,1) and h[2:5]==(0,0,8) and status[0]==6 and h[5]==len(raw)
at=256+128*nb;blocks=raw[256:at]
def take(n):
 global at
 value=np.frombuffer(raw,dtype='<f4',count=n,offset=at).copy();at+=4*n;return value
M=take(nv*nv);J=take(rows*nv);bias=take(rows);free=take(nv);warm=take(nv);warmP=take(rows);failed=take(nv);imp=take(rows);rowV=take(rows);grad=take(nv)
packed=struct.pack('<8I',0x53504331,1,nv,rows,nb,20,0,0)+blocks+b''.join(x.tobytes() for x in [M,J,bias,free,failed])
(B/'failed-operator.bin').write_bytes(packed)
c=ref.parse(packed);cal=ref.calibrate();q=ref.evaluate(c,failed.astype(float))
v,solution,history=ref.reference(c);rounded=ref.evaluate(c,v.astype('<f4').astype(float))
av=c['A']@(failed.astype(float)-c['free']);jp=c['J'].T@imp.astype(float)
published=float(np.max(np.abs(av-jp))/(1+np.max(np.abs(av))+np.max(np.abs(jp))))
np.asarray(v,dtype='<f8').tofile(B/'reference-velocity.f64')
obs=dict(owners_passed=True,archived_physical_files_exact_per_repeat=len(matched['api-0']),strict_capture_exact=True,
 status6_first_witness_matches_public=True,status6_repeated_capture_exact=True,independent_reference_updates=len(history)-1,
 independent_reference_within_20_updates=len(history)-1<=20,rounded_reference_meets_original_gates=rounded['res']<=d[16] and rounded['feas']<=d[17])
analysis=dict(capture_sha256=sha(O/'api-0-capture.bin'),header=list(h),dispatch=list(d),status=list(status),calibration=cal,
 independent_failed_projected_residual=q['res'],independent_failed_published_residual=published,native_failed_residual=status[12],
 independent_failed_feasibility=q['feas'],initial_relative_projected_vs_native=abs(q['res']-status[12])/max(q['res'],status[12]),
 native_gradient_vs_independent_abs_error=float(np.max(np.abs(grad.astype(float)-q['g']))),
 native_row_velocity_vs_independent_abs_error=float(np.max(np.abs(rowV.astype(float)-q['w']))),
 reference_residual_history=history,reference_feasibility=solution['feas'],reference_rounded_residual=rounded['res'],reference_rounded_feasibility=rounded['feas'],
 reference_velocity_distance_from_failed=float(np.max(np.abs(v-failed))),physical_files_exact=matched)
write(A/'analysis.json',analysis);write(A/'result.json',dict(observations=obs,native_commands=len(execution),goal_complete=False,
 independent_analysis_only=True,policy_or_profile_promoted=False))
print((A/'result.json').read_text(),flush=True);print((A/'analysis.json').read_text(),flush=True)
