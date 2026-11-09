from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-four-expert-native-integration-20261109';B=R/'build/crow-four-expert-native-integration-20261109';O=R/'build/crow-four-native-observer-20261109'
prior=R/'artifacts/crow-four-expert-eval-20261109/stage4-independent-20-landing-baseline/reserved-final-c-seed-20860117';d=json.loads((prior/'results.json').read_text());cmd=d['argv'].copy()
for k,v in [('--birdflow-journey-variant','v15-triple-waypoint-expert-navigation'),('--metallib',str(R/'build/shaders/MetalRobo.metallib')),('--policy-pack',str(B/'landing.policypack')),('--base-policy-pack',str(B/'route.policypack'))]:cmd[cmd.index(k)+1]=v
runs={}
for mode in ['ordinary','observer']:
 lib=R/'build/lib' if mode=='ordinary' else O
 trace=B/(mode+'-parity.tsv');cmd[cmd.index('--state-trace')+1]=str(trace)
 env={**os.environ,'DYLD_LIBRARY_PATH':str(lib),'PYTHONDONTWRITEBYTECODE':'1'}
 if mode=='observer':env.update(CROW_CONTACT_TRACE='1',CROW_SUPPORT_TRACE='1',CROW_CLEARANCE_TRACE='1')
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
 (A/(mode+'-parity.stdout')).write_text(p.stdout)
 with (A/(mode+'-parity.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 with trace.open() as f:
  f.readline();rows={int(x[0]):[float(v) for v in x[1:24]] for line in f for x in [line.strip().split('\t')] if len(x)>24}
 runs[mode]={'q':rows,'first':json.loads(p.stdout)['first_episode_navigation'],'stderr':p.stderr,'library_sha256':hashlib.sha256((lib/'libmetalrobo.dylib').read_bytes()).hexdigest()}
a,b=runs['ordinary'],runs['observer'];end=a['first']['steps_by_environment'][1]
keys=sorted(k for k in a['q'] if k<=end and k in b['q']);assert len(keys)==end
maxerr=max(abs(i-j) for k in keys for i,j in zip(a['q'][k],b['q'][k]))
def samples(lines):
 return {(z['lane'],z['step']):z for l in lines if l.startswith('CROW_FLIGHT_SAMPLE ') for z in [json.loads(l.split(' ',1)[1])]}
with gzip.open(prior/'rollout.stderr.gz','rt') as f:old=samples(f)
new=samples(b['stderr'].splitlines());oldends=d['first_episode']['steps_by_environment']
samplekeys=[k for k in old if k[1]<=oldends[k[0]]]
bad=[k for k in samplekeys if k not in new or old[k]['q']!=new[k]['q'] or old[k]['v']!=new[k]['v']]
records=[json.loads(l.split(' ',1)[1]) for l in b['stderr'].splitlines() if l.startswith('CROW_CLEARANCE_SAMPLE ')]
for lane,e in enumerate(b['first']['steps_by_environment']):assert any(z['lane']==lane and z['end_step'] in (e,e+1) for z in records),(lane,e)
report={'observer_first_episode_equal':a['first']==b['first'],'observer_q_exact':all(a['q'][k]==b['q'][k] for k in keys),'observer_q_max_abs_difference':maxerr,'observer_lane1_steps_compared':len(keys),'experimental_first_episode_equal':d['first_episode']==b['first'],'experimental_samples_compared':len(samplekeys),'experimental_qv_mismatches':len(bad),'experimental_first_mismatches':bad[:10],'ordinary_library_sha256':a['library_sha256'],'observer_library_sha256':b['library_sha256'],'shader_sha256':hashlib.sha256((R/'build/shaders/MetalRobo.metallib').read_bytes()).hexdigest(),'argv':cmd,'scope':'production observer vs ordinary and experimental physical comparison on one development course; clearance is control-endpoint sampled'}
(A/'parity-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['observer_first_episode_equal','observer_q_exact','observer_lane1_steps_compared','experimental_first_episode_equal','experimental_qv_mismatches']}),flush=True)
assert report['observer_first_episode_equal'] and report['observer_q_exact']
