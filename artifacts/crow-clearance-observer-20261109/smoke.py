from pathlib import Path
import gzip,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-clearance-observer-20261109';B=R/'build/crow-clearance-observer-20261109'
cmd=json.loads((R/'artifacts/crow-latched-joint-dev-20261109/training/route-results.json').read_text())['argv']
for key,value in [('--steps','500'),('--envs','4'),('--seed','20270120'),('--metallib',str(R/'build/crow-stage-selected-policy-20261109/stage-selected.metallib')),('--policy-pack',str(R/'build/crow-stage-selected-policy-20261109/landing-actor.policypack')),('--state-trace',str(B/'smoke.tsv'))]:
 cmd[cmd.index(key)+1]=value
cmd+=['--base-policy-pack',str(R/'build/crow-deterministic-route-training-20261109/frozen-route-actor.policypack')]
env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
env.pop('CROW_SUPPORT_TRACE',None);env.pop('CROW_CLEARANCE_TRACE',None)
runs={}
for mode in ['disabled','enabled']:
 e=env.copy()
 if mode=='enabled':e['CROW_CLEARANCE_TRACE']='1'
 start=time.monotonic();p=subprocess.run(cmd,cwd=R,env=e,text=True,capture_output=True,timeout=360);elapsed=time.monotonic()-start
 if p.returncode:raise RuntimeError(p.stderr[-4000:])
 (A/(mode+'.stdout')).write_text(p.stdout)
 with (A/(mode+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 samples={(x['lane'],x['step']):x for l in p.stderr.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(l.split(' ',1)[1])]}
 clearance=[json.loads(l.split(' ',1)[1]) for l in p.stderr.splitlines() if l.startswith('CROW_CLEARANCE_SAMPLE ')]
 runs[mode]={'samples':samples,'clearance':clearance,'seconds':elapsed}
a,b=(runs[x] for x in ['disabled','enabled'])
bad=[k for k,x in a['samples'].items() if k not in b['samples'] or x['q']!=b['samples'][k]['q'] or x['v']!=b['samples'][k]['v']]
assert len(a['samples'])==20 and not bad and len(b['clearance'])==4
assert all(x['minimum_m']>=0 and x['sampling']=='accepted_control_steps' for x in b['clearance'])
report={'steps':500,'lanes':4,'matched_physics_samples':len(a['samples']),'mismatches':len(bad),'clearance':b['clearance'],'seconds':{k:v['seconds'] for k,v in runs.items()},'scope':'endpoint-sampled geometry, not continuous substep minimum'}
(A/'smoke-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
