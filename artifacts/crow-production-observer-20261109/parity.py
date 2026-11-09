from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-production-observer-20261109';B=R/'build/crow-production-observer-20261109'
cmd=json.loads((R/'artifacts/crow-terminal-specialist-paired-eval-20261109/training/results.json').read_text())['argv'].copy()
for k,v in [('--birdflow-journey-variant','v15-hierarchical-navigation'),('--metallib',str(R/'build/shaders/MetalRobo.metallib')),('--seed','20270203')]:cmd[cmd.index(k)+1]=v
runs={}
for mode in ['ordinary','observer']:
 lib=R/'build/lib' if mode=='ordinary' else B
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
 runs[mode]={'q':rows,'first':json.loads(p.stdout)['first_episode_navigation'],'library_sha256':hashlib.sha256((lib/'libmetalrobo.dylib').read_bytes()).hexdigest(),'stderr':p.stderr}
a,b=runs['ordinary'],runs['observer'];end=a['first']['steps_by_environment'][1]
keys=sorted(k for k in a['q'] if k<=end and k in b['q'])
assert len(keys)==end,(len(keys),end)
maxerr=max(abs(i-j) for k in keys for i,j in zip(a['q'][k],b['q'][k]))
records=[json.loads(l.split(' ',1)[1]) for l in b['stderr'].splitlines() if l.startswith('CROW_CLEARANCE_SAMPLE ')]
final=[]
for lane,end in enumerate(a['first']['steps_by_environment']):
 matches=[z for z in records if z['lane']==lane and z['end_step'] in (end,end+1)]
 assert matches,(lane,end)
 z=next((x for x in matches if x['end_step']==end),matches[0]);z['first_episode_end_step']=end;final.append(z)
report={'argv':cmd,'first_episode_equal':a['first']==b['first'],'steps_compared':len(keys),'q_max_absolute_difference':maxerr,'q_exact':all(a['q'][k]==b['q'][k] for k in keys),'lane_minima':final,'ordinary_library_sha256':a['library_sha256'],'observer_library_sha256':b['library_sha256'],'shader_sha256':hashlib.sha256((R/'build/shaders/MetalRobo.metallib').read_bytes()).hexdigest(),'scope':'same production shader; all lane outcomes and lane 1 state trajectory; clearance sampled at accepted control-step endpoints, not continuous'}
(A/'parity-results.json').write_text(json.dumps(report,indent=2)+'\n')
print('first episodes equal',report['first_episode_equal'],'q exact',report['q_exact'],'steps',len(keys),'clearance lanes',len(final),flush=True)
assert report['first_episode_equal'] and report['q_exact']
