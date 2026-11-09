from pathlib import Path
import difflib,gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
walk='''        // Experimental gait ends before the launch crouch takes control.
        if (fixedFlightController && state.episode.z == 10u && action >= 7u && action <= 12u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            const uint leg=(action-7u)/3u, joint=(action-7u)%3u;
            const float support=leg==0u ? state.nutrition.y : state.nutrition.x;
            const float carrier=sin(kTwoPi*seconds/0.50f)*(leg==0u ? 1.0f : -1.0f);
            const float swing=seconds >= 2.0f && seconds < 4.3f && support > 0.05f ? carrier : 0.0f;
            const float lift=max(swing,0.0f);
            if (seconds >= 2.0f && seconds < 4.3f)
                navigationTeacherRequested=joint==0u ? -0.30f*swing : joint==1u ? 0.40f*lift : -0.15f*lift;
        }
'''
for wings,arm in [('off','control'),('off','push'),('on','control'),('on','push')]:
 old=('launch-perturb-' if wings=='off' else 'launch-wing-on-')+arm;name='walk-launch-wings-'+wings+'-'+arm;src_text=(B/(old+'.metal')).read_text();mark='        // Experimental launch leg targets; accepted ground load gates each leg.\n' if arm=='push' else '        // Keep the training-only teacher active through the final target.\n';assert src_text.count(mark)==1;src_text=src_text.replace(mark,walk+mark)
 src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(src_text);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff((R/'src/metal/LocomotionTask.metal').read_text().splitlines(True),src_text.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/'+name+'.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.load(open(A/(old+'-bindings.json')))['argv'];
 for k,v in [('--metallib',str(lib)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'walk-launch-plan.md'),'patch_sha256':h(A/(name+'.patch')),'shader_sha256':h(lib),'observer_sha256':h(C/'libmetalrobo.dylib'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 print(name,'finished',flush=True)
