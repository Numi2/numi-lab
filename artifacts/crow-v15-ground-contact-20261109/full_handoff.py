from pathlib import Path
import gzip,hashlib,json,math,os,subprocess,sys
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';C=B/'late-observer';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();base=json.loads((A/'handoff-reserved-final-i-bindings.json').read_text())['argv'];lib=B/'walk-handoff.metallib';assert h(lib)==json.loads((A/'handoff-reserved-final-i-bindings.json').read_text())['shader_sha256']
def run(name,course,seed,steps,trace):
 cmd=base.copy()
 for k,v in [('--birdflow-navigation-course',course),('--seed',str(seed)),('--steps',str(steps)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'PYTHONDONTWRITEBYTECODE':'1'}
 if trace:env.update(CROW_CONTACT_TRACE='1',CROW_SUPPORT_TRACE='1')
 else:
  env.pop('CROW_CONTACT_TRACE',None);env.pop('CROW_SUPPORT_TRACE',None)
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'full-handoff-plan.md'),'shader_sha256':h(lib),'observer_sha256':h(C/'libmetalrobo.dylib'),'trace':trace},indent=2)+'\n')
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=400);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+': '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 return json.loads(raw)['payload']['frames'],json.loads(p.stdout),p.stderr
# Instrumentation test is an exact matched comparison of the complete accepted states.
a,_,_=run('late-observer-parity-off','reserved-final-i',4038889268,350,False)
b,_,_=run('late-observer-parity-on','reserved-final-i',4038889268,350,True)
parity={'frames':len(a),'complete_replay_frames_equal':a==b};(A/'late-observer-parity.json').write_text(json.dumps(parity,indent=2)+'\n');print('observer parity',parity,flush=True)
if a!=b:raise RuntimeError('observer changed accepted states')
if '--parity-only' in sys.argv:sys.exit(0)
for course,seed in [('reserved-final-i',4038889268),('reserved-final-j',3193706597)]:
 name='full-handoff-'+course;frames,data,stderr=run(name,course,seed,3000,True);perlane=[{} for _ in range(2)]
 for line in stderr.splitlines():
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  s=json.loads(line.split(' ',1)[1]);lane=s['lane'];step=s['step'];up=0;foot=False;other=False
  for c in s['contacts']:
   a,b=c['a_name'],c['b_name']
   if 'crow_course_perch' not in (a,b):continue
   othername=b if a=='crow_course_perch' else a
   if c['impulses'][0]<=0:continue
   if othername in ('crow_left_foot','crow_right_foot'):
    foot=True;up+=(-1 if a==othername else 1)*c['normal'][2]*c['impulses'][0]
   else:other=True
  v=s['v'];valid=foot and not other and up>0 and math.hypot(v[0],v[1])<=.12 and abs(v[2])<=.12
  perlane[lane][step]={'valid':valid,'upward_endpoint_impulse_ns':up,'q':s['q'],'v':v}
 rows=[]
 for lane in range(2):
  checks=[perlane[lane].get(step) for step in range(2976,3001)];valid=sum(bool(x and x['valid']) for x in checks);rows.append({'lane':lane,'terminal_supported_foot_only_samples':valid,'terminal_support_gate':valid==25,'terminal_sample_count':sum(x is not None for x in checks),'first_physical_failure':data.get('first_episode_navigation',{}).get('physical_failure_by_environment',[None,None])[lane]})
 report={'course':course,'seed':seed,'lanes':rows,'failed_environment_steps':data.get('failed_environment_steps'),'first_episode_navigation':data.get('first_episode_navigation'),'observed_replay_frames':len(frames),'scope':'development layout; support on flat perch is not grasp'};(A/(name+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(report),flush=True)
