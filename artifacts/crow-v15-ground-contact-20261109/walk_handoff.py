from pathlib import Path
import difflib,gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';base=(B/'feedback_live.metal').read_text();s=base
for old,new in [('fixedFlightController && state.episode.z == 1u && action >= 7u','fixedFlightController && state.episode.z == 10u && action >= 7u'),('state.episode.z == 1u && program.counts0.w >= 2u','state.episode.z == 10u && program.counts0.w >= 2u')]:
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
(A/'walk-handoff.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/feedback_live.metal',tofile='b/walk-handoff.metal')))
src=B/'walk-handoff.metal';air=B/'walk-handoff.air';lib=B/'walk-handoff.metallib';src.write_text(s)
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for course,seed in [('reserved-final-i',4038889268),('reserved-final-j',20261124)]:
 name='handoff-'+course;cmd=json.loads((A/'bindings.json').read_text())['argv']
 for k,v in [('--metallib',str(lib)),('--birdflow-navigation-course',course),('--seed',str(seed)),('--state-trace',str(B/(name+'.tsv')))]:cmd[cmd.index(k)+1]=v
 cmd+=['--crow-replay-pack',str(B/(name+'.json'))]
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'walk-handoff-plan.md'),'patch_sha256':h(A/'walk-handoff.patch'),'shader_sha256':h(lib),'base_shader_binding':'feedback_live-bindings.json','flight_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+': '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 frames=json.loads(raw)['payload']['frames'];d=json.loads(p.stdout);lanes=[{} for _ in range(2)]
 for line in p.stderr.splitlines():
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  sample=json.loads(line.split(' ',1)[1]);entry=lanes[sample['lane']].setdefault(sample['step'],{'feet':set(),'nonfoot':False})
  for c in sample['contacts']:
   a,b=c['a_name'],c['b_name']
   if 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
   foot=next((x for x in (a,b) if x in ('crow_left_foot','crow_right_foot')),None)
   if foot and (-1 if a==foot else 1)*c['normal'][2]*c['impulses'][0]>0:entry['feet'].add(foot)
   elif not foot:entry['nonfoot']=True
 rows=[]
 for lane,ss in enumerate(lanes):
  row={'lane':lane,'left_only_samples':sum(x['feet']=={'crow_left_foot'} for x in ss.values()),'right_only_samples':sum(x['feet']=={'crow_right_foot'} for x in ss.values()),'last_ground_sample_step':max(ss,default=None),'nonfoot_contact_steps':sum(x['nonfoot'] for x in ss.values())};rows.append(row)
 a,b=frames[99]['q'],frames[249]['q'];disp=math.hypot(b[0]-a[0],b[1]-a[1]);report={'course':course,'seed':seed,'lanes':rows,'observed_walk_displacement_m':disp,'observed_final_height_m':frames[-1]['q'][2],'failed_environment_steps':d.get('failed_environment_steps'),'first_episode_navigation':d.get('first_episode_navigation'),'scope':'development handoff only'};(A/(name+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps({k:v for k,v in report.items() if k!='first_episode_navigation'}),flush=True)
