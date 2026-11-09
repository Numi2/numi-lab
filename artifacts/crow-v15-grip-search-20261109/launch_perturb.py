from pathlib import Path
import difflib,gzip,hashlib,json,subprocess,os,math
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';base=(B/'crouch85-wings-off.metal').read_text()
needle='''        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;
'''
assert base.count(needle)==1
inject='''        // Initial-state perturbation only; no mid-episode state edits or injected force.
        if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && curriculum == 10u) {
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
base=base.replace(needle,inject+needle)
start='        // Experimental launch leg targets; accepted ground load gates each leg.\n';end='        // Keep the training-only teacher active through the final target.\n'
assert base.count(start)==1 and base.count(end)==1
control=base[:base.index(start)]+base[base.index(end):]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,src_text in [('launch-perturb-control',control),('launch-perturb-push',base)]:
 (A/(name+'.patch')).write_text(''.join(difflib.unified_diff((R/'src/metal/LocomotionTask.metal').read_text().splitlines(True),src_text.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/'+name+'.metal')))
 src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(src_text)
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.loads((R/'artifacts/crow-v15-ground-contact-20261109/bindings.json').read_text())['argv']
 for k,v in [('--metallib',str(lib)),('--envs','8'),('--seed','20261126'),('--birdflow-navigation-course','training'),('--state-trace',str(B/(name+'.tsv')))]:cmd[cmd.index(k)+1]=v
 cmd+=['--crow-replay-pack',str(B/(name+'.json'))]
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'launch-perturb-plan.md'),'patch_sha256':h(A/(name+'.patch')),'shader_sha256':h(lib),'observer_sha256':h(B/'materialized/libmetalrobo.dylib'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'materialized'),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 for suffix,raw in [('stderr.gz',p.stderr.encode()),('replay.json.gz',(B/(name+'.json')).read_bytes() if p.returncode==0 else b'')]:
  with (A/(name+'.'+suffix)).open('wb') as f:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-3000:])
 print(name,'finished',flush=True)
