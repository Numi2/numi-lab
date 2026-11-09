from pathlib import Path
import difflib,gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';original=(R/'src/metal/LocomotionTask.metal').read_text()
def once(s,a,b):
 if s.count(a)!=1:raise RuntimeError(f'marker count {s.count(a)}: {a[:80]}')
 return s.replace(a,b)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,push,suppress in [('fast-wings-off-push',True,True),('fast-wings-normal-push',True,False)]:
 s=original
 if push:
  mark='        // Keep the training-only teacher active through the final target.\n'
  patch='''        // Experimental launch leg targets; accepted ground load gates each leg.
        if (fixedFlightController && state.episode.z == 10u && action >= 7u && action <= 12u) {
            const float seconds = float(state.episode.x)*dispatch.timing.x;
            const uint leg=(action-7u)/3u, joint=(action-7u)%3u;
            const float footLoad=leg==0u ? state.nutrition.x : state.nutrition.y;
            if (seconds >= 4.90f && seconds < 5.2f && footLoad > 0.05f) {
                const float ramp=smoothstep(4.90f,4.98f,seconds);
                const float extended=joint==0u ? 0.10f : joint==1u ? -0.36f : 0.18f;
                navigationTeacherRequested=mix(navigationTeacherRequested,extended,ramp);
            }
        }
'''
  s=once(s,mark,patch+mark)
 if suppress:
  mark='''        const float avianWingAmplitude =
            fixedFlightController && state.threatTeacher.w >= 1.5f &&
'''
  replacement='''        const float avianWingAmplitude =
            fixedFlightController && state.episode.z == 10u && float(state.episode.x)*dispatch.timing.x >= 4.4f && float(state.episode.x)*dispatch.timing.x < 5.4f && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.5f &&
'''
  s=once(s,mark,replacement)
 mark='    taskStates[environment] = state;\n\n    const uint transitionIndex =\n'
 inject='''    // Accepted foot force diagnostic feedback for this isolated band-10 bench.
    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && state.episode.z == 10u && program.counts0.w >= 2u) {
        state.nutrition.xy=float2(compactContact[compactBase+contactGroups[0].members.w],compactContact[compactBase+contactGroups[1].members.w]);
    }
'''
 s=once(s,mark,inject+mark)
 (A/(name+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/'+name+'.metal')))
 src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(s)
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.loads((A/'bindings.json').read_text())['argv']
 for k,v in [('--metallib',str(lib)),('--state-trace',str(B/(name+'.tsv')))]:cmd[cmd.index(k)+1]=v
 cmd += ['--crow-replay-pack',str(B/(name+'.json'))]
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'pushoff-fast-plan.md'),'patch_sha256':h(A/(name+'.patch')),'shader_sha256':h(lib),'observer_sha256':h(B/'libmetalrobo.dylib'),'frozen_flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal'),'baseline_bindings':'bindings.json'},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 frames=json.loads(raw)['payload']['frames'];data=json.loads(p.stdout);loads={};nonfoot=0;peak_impulse=0.0
 for line in p.stderr.splitlines():
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  sample=json.loads(line.split(' ',1)[1])
  if sample['lane']!=1:continue
  flags=loads.setdefault(sample['step'],set())
  for c in sample['contacts']:
   a,b=c['a_name'],c['b_name']
   if 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
   foot=next((x for x in (a,b) if x in ('crow_left_foot','crow_right_foot')),None)
   if foot and (-1 if a==foot else 1)*c['normal'][2]*c['impulses'][0]>0:
    flags.add(foot)
    if 230<=sample['step']<=270:peak_impulse=max(peak_impulse,c['impulses'][0])
   elif not foot:nonfoot+=1
 window=[f for f in frames if 230<=f['step']<=270]
 report={'name':name,'frames':len(frames),'max_vz_4_6_to_5_4_m_s':max(f['v'][2] for f in window),'max_z_4_6_to_5_4_m':max(f['q'][2] for f in window),'maximum_sampled_foot_normal_impulse_Ns':peak_impulse,'sampled_both_feet_unloaded_4_6_to_5_4':sum(not loads.get(f['step'],set()) for f in window),'nonfoot_ground_contact_rows':nonfoot,'failed_environment_steps':data.get('failed_environment_steps'),'first_episode_navigation':data.get('first_episode_navigation'),'scope':'endpoint reports; peak impulse is not time integral'}
 (A/(name+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps({k:v for k,v in report.items() if k!='first_episode_navigation'}),flush=True)
