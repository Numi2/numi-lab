from pathlib import Path
import difflib,gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';base=(B/'feedback_live.metal').read_text();needle='''        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;
'''
assert base.count(needle)==1
inject='''        // Isolated perturbation screen: no forces or kinematic replay.
        if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && curriculum == 1u) {
            const float roll=0.06f*randomSigned(dispatch,environment,episode,0u,8990u);
            const float4 orientation=float4(resetQ[qBase+program.root.z+3u],resetQ[qBase+program.root.z+4u],resetQ[qBase+program.root.z+5u],resetQ[qBase+program.root.z+6u]);
            const float4 qroll=quaternionProduct(orientation,float4(sin(0.5f*roll),0.0f,0.0f,cos(0.5f*roll)));
            for (uint k=0u;k<4u;++k) resetQ[qBase+program.root.z+3u+k]=qroll[k];
            for (uint a=7u;a<=12u;++a) {
                const MRTaskActionBindingGPU binding=actions[a];
                resetQ[qBase+binding.indices.z]=clamp(resetQ[qBase+binding.indices.z]+0.04f*randomSigned(dispatch,environment,episode,0u,9000u+a),binding.parameters.y,binding.parameters.z);
            }
            for (uint k=0u;k<2u;++k) resetV[vBase+program.root.w+k]+=0.025f*randomSigned(dispatch,environment,episode,0u,9020u+k);
        }
'''
s=base.replace(needle,inject+needle);(A/'gait-perturb.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/feedback_live.metal',tofile='b/gait-perturb.metal')))
src=B/'gait-perturb.metal';air=B/'gait-perturb.air';lib=B/'gait-perturb.metallib';src.write_text(s)
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cmd=json.loads((A/'feedback_live-bindings.json').read_text())['argv']
for k,v in [('--metallib',str(lib)),('--envs','8'),('--seed','20261126'),('--birdflow-navigation-course','training'),('--state-trace',str(B/'gait-perturb.tsv')),('--crow-replay-pack',str(B/'gait-perturb.json'))]:cmd[cmd.index(k)+1]=v
(A/'gait-perturb-bindings.json').write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'gait-perturb-plan.md'),'patch_sha256':h(A/'gait-perturb.patch'),'shader_sha256':h(lib),'base_shader_binding':'feedback_live-bindings.json'},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/'gait-perturb.stdout').write_text(p.stdout)
with (A/'gait-perturb.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-3000:])
raw=(B/'gait-perturb.json').read_bytes()
with (A/'gait-perturb.replay.json.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
data=json.loads(p.stdout);lanes=[{} for _ in range(8)]
for line in p.stderr.splitlines():
 if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
 s=json.loads(line.split(' ',1)[1]);lane=s['lane'];entry=lanes[lane].setdefault(s['step'],{'q':s['q'],'feet':set(),'nonfoot':False})
 for c in s['contacts']:
  a,b=c['a_name'],c['b_name']
  if 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
  foot=next((x for x in (a,b) if x in ('crow_left_foot','crow_right_foot')),None)
  if foot and (-1 if a==foot else 1)*c['normal'][2]*c['impulses'][0]>0:entry['feet'].add(foot)
  elif not foot:entry['nonfoot']=True
rows=[]
for lane,ss in enumerate(lanes):
 steps=sorted(ss);a=ss[steps[0]]['q'] if steps else None;b=ss[steps[-1]]['q'] if steps else None
 def tilt(q):
  x,y,z,w=q[3:7];return math.acos(max(-1,min(1,1-2*(x*x+y*y))))
 foot=('crow_left_foot','crow_right_foot');d=math.hypot(b[0]-a[0],b[1]-a[1]) if a else None
 row={'lane':lane,'first_sample':steps[0] if steps else None,'last_sample':steps[-1] if steps else None,'first_quaternion_xyzw':a[3:7] if a else None,'first_leg_joint_positions':a[14:20] if a else None,'observed_net_displacement_m':d,'max_sampled_tilt_radians':max((tilt(x['q']) for x in ss.values()),default=None),'left_only_samples':sum(x['feet']=={foot[0]} for x in ss.values()),'right_only_samples':sum(x['feet']=={foot[1]} for x in ss.values()),'nonfoot_ground_contact_steps':sum(x['nonfoot'] for x in ss.values())}
 row['necessary_gate_pass']=bool(steps and steps[0]<=10 and steps[-1]==350 and d>=0.05 and row['left_only_samples'] and row['right_only_samples'] and not row['nonfoot_ground_contact_steps']);rows.append(row)
report={'lanes':rows,'failed_environment_steps':data.get('failed_environment_steps'),'first_episode_navigation':data.get('first_episode_navigation'),'scope':'perturbation screen, not held-out biological qualification'};(A/'gait-perturb-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
