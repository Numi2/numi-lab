from pathlib import Path
import difflib,gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old='''        // Experimental launch leg targets; accepted ground load gates each leg.
        if (fixedFlightController && state.episode.z == 10u && action >= 7u && action <= 12u) {
            const float seconds = float(state.episode.x)*dispatch.timing.x;
            const uint leg=(action-7u)/3u, joint=(action-7u)%3u;
            const float footLoad=leg==0u ? state.nutrition.x : state.nutrition.y;
            if (seconds >= 4.3f && seconds < 5.2f && footLoad > 0.05f) {
                const float crouched=joint==0u ? -0.85f : joint==1u ? 1.00f : -0.50f;
                const float extended=joint==0u ? 0.10f : joint==1u ? -0.70f : 0.18f;
                navigationTeacherRequested=seconds<4.96f
                    ? mix(navigationTeacherRequested,crouched,smoothstep(4.3f,4.55f,seconds))
                    : mix(crouched,extended,smoothstep(4.96f,5.00f,seconds));
            }
        }
'''
new='''        // Symmetric preparation; extension uses both accepted foot loads.
        if (fixedFlightController && state.episode.z == 10u && action >= 7u && action <= 12u) {
            const float seconds=float(state.episode.x)*dispatch.timing.x;
            const uint joint=(action-7u)%3u;
            if (seconds >= 4.3f && seconds < 5.2f) {
                const float crouched=joint==0u ? -0.85f : joint==1u ? 1.00f : -0.50f;
                const float extended=joint==0u ? 0.10f : joint==1u ? -0.70f : 0.18f;
                const bool bilateral=state.nutrition.x>0.05f && state.nutrition.y>0.05f;
                navigationTeacherRequested=seconds<4.96f
                    ? mix(navigationTeacherRequested,crouched,smoothstep(4.3f,4.55f,seconds))
                    : bilateral ? mix(crouched,extended,smoothstep(4.96f,5.00f,seconds)) : crouched;
            }
        }
'''
for wings,arm in [('off','control'),('off','push'),('on','control'),('on','push')]:
 oldname='walk-launch-wings-'+wings+'-'+arm;name='symmetric-launch-wings-'+wings+'-'+arm;src_text=(B/(oldname+'.metal')).read_text();mark='        // Experimental gait ends before the launch crouch takes control.\n';end='        // Experimental launch leg targets;' if arm=='push' else '        // Keep the training-only teacher active through the final target.'
 i=src_text.index(mark);j=src_text.index(end,i);part=src_text[i:j];assert part.count('4.3f')==2;src_text=src_text[:i]+part.replace('4.3f','4.0f')+src_text[j:]
 if arm=='push':assert src_text.count(old)==1;src_text=src_text.replace(old,new)
 src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(src_text);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff((R/'src/metal/LocomotionTask.metal').read_text().splitlines(True),src_text.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/'+name+'.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.load(open(A/(oldname+'-bindings.json')))['argv'];
 for k,v in [('--metallib',str(lib)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'symmetric-launch-plan.md'),'patch_sha256':h(A/(name+'.patch')),'shader_sha256':h(lib),'observer_sha256':h(C/'libmetalrobo.dylib'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 print(name,'finished',flush=True)
