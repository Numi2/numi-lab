from pathlib import Path
import difflib,gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';B=R/'build/crow-v15-contact-slip-20261109';O=R/'artifacts/crow-v15-ground-contact-20261109';C=R/'build/crow-v15-ground-contact-20261109';base=(C/'feedback_live.metal').read_text();h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def evaluate(payload,stderr,data):
 frames={f['step']:f for f in payload['frames']};feet=('crow_left_foot','crow_right_foot');stats={n:[] for n in feet};loads={};nonfoot=set()
 for line in stderr.splitlines():
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  sample=json.loads(line.split(' ',1)[1]);step=sample['step']
  if sample['lane']!=1 or not 101<=step<=250:continue
  f=frames[step];stride=payload['body_state_stride'];flags=loads.setdefault(step,set())
  for c in sample['contacts']:
   a,b=c['a_name'],c['b_name'];name=next((v for v in (a,b) if v in feet),None)
   if 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
   if not name:nonfoot.add(step);continue
   normal=c['normal'];up=(-1 if a==name else 1)*normal[2]*c['impulses'][0]
   if up<=0:continue
   flags.add(name)
   if up<=.0001:continue
   idx=payload['body_names'].index(name);st=f['body_states'][idx*stride:(idx+1)*stride];v=st[7:10];w=st[10:13];r=[c['point'][i]-st[i] for i in range(3)];cross=[w[1]*r[2]-w[2]*r[1],w[2]*r[0]-w[0]*r[2],w[0]*r[1]-w[1]*r[0]];vc=[v[i]+cross[i] for i in range(3)];dot=sum(vc[i]*normal[i] for i in range(3))/sum(x*x for x in normal);stats[name].append(math.sqrt(sum((vc[i]-dot*normal[i])**2 for i in range(3))))
 dist={}
 for n,x in stats.items():
  x.sort();dist[n]={'sample_count':len(x),'median_m_s':statistics.median(x) if x else None,'p95_m_s':x[math.ceil(.95*len(x))-1] if x else None,'max_m_s':max(x) if x else None}
 a,b=frames[100]['q'],frames[250]['q'];travel=math.hypot(b[0]-a[0],b[1]-a[1]);first=data['first_episode_navigation'];rows={'window_net_displacement_m':travel,'feet':dist,'left_only_steps':sum(x=={feet[0]} for x in loads.values()),'right_only_steps':sum(x=={feet[1]} for x in loads.values()),'nonfoot_steps':len(nonfoot),'failed_environment_steps':data['failed_environment_steps'],'physical_failure_count':first['physical_failure_count']};rows['necessary_screen_pass']=bool(travel>=.03 and rows['left_only_steps'] and rows['right_only_steps'] and not nonfoot and not rows['failed_environment_steps'] and not rows['physical_failure_count'] and all(x['p95_m_s'] is not None and x['p95_m_s']<=.04 for x in dist.values()));return rows
old=json.load(gzip.open(O/'feedback_live.replay.json.gz','rt'))['payload'];stderr=gzip.open(A/'rollout.stderr.gz','rt').read();baseline=evaluate(old,stderr,json.loads((A/'rollout.stdout').read_text()));(A/'stance-baseline-summary.json').write_text(json.dumps(baseline,indent=2)+'\n');print('baseline',json.dumps(baseline),flush=True)
for name,gain in [('stance-gain-05',0.5),('stance-gain-15',1.5),('stance-gain-30',3.0)]:
 s=base
 old="            navigationTeacherRequested = joint == 0u ? -0.30f*swing : joint == 1u ? 0.40f*lift : -0.15f*lift;\n"
 new=old+"            const float ownLoad = leg == 0u ? state.nutrition.x : state.nutrition.y;\n"+f"            if (joint == 0u && carrier < 0.0f && seconds >= 2.0f && seconds < 5.0f && ownLoad > 0.05f) navigationTeacherRequested += {gain:.2f}f * clamp(state.nutrition[2u+leg], -0.15f, 0.15f);\n"
 assert s.count(old)==1;s=s.replace(old,new)
 old='        state.nutrition.xy = float2(compactContact[compactBase+contactGroups[0].members.w], compactContact[compactBase+contactGroups[1].members.w]);\n'
 new=old+"""        // Previous solved center of pressure and body state estimate the stance patch velocity.
        for (uint foot=0u;foot<2u;++foot) {
            const MRTaskContactGroupGPU group=contactGroups[foot];
            const uint metric=compactBase+group.members.w;
            const MRBodyStateGPU body=bodyStates[bodyBase+group.reference.x];
            const float3 r=rotate(body.orientation,float3(compactContact[metric+4u],compactContact[metric+5u],-0.008f));
            const float3 velocity=body.linearVelocityAndInverseMass.xyz+cross(body.angularVelocity.xyz,r);
            state.nutrition[foot+2u]=dot(velocity.xy,yawBasis);
        }
"""
 assert s.count(old)==1;s=s.replace(old,new)
 (A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/feedback_live.metal',tofile='b/'+name+'.metal')));src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(s)
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.loads((A/'bindings.json').read_text())['argv']
 for k,v in [('--metallib',str(lib)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'stance-feedback-plan.md'),'shader_sha256':h(lib),'patch_sha256':h(A/(name+'.patch')),'observer_sha256':h(B/'libmetalrobo.dylib')},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+': '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 report=evaluate(json.loads(raw)['payload'],p.stderr,json.loads(p.stdout));(A/(name+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(report),flush=True)
