from pathlib import Path
import difflib,gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-grip-search-20261109';B=R/'build/crow-v15-grip-search-20261109';C=B/'complete-observer';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm in ['control','push']:
 old='launch-perturb-'+arm;name='launch-wing-on-'+arm;src_text=(B/(old+'.metal')).read_text()
 before='''            fixedFlightController && state.episode.z == 10u && float(state.episode.x)*dispatch.timing.x >= 4.4f && float(state.episode.x)*dispatch.timing.x < 5.4f && binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f : fixedFlightController && state.threatTeacher.w >= 1.5f &&
''';after='''            fixedFlightController && state.threatTeacher.w >= 1.5f &&
'''
 assert src_text.count(before)==1;src_text=src_text.replace(before,after)
 src=B/(name+'.metal');air=B/(name+'.air');lib=B/(name+'.metallib');src.write_text(src_text);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff((R/'src/metal/LocomotionTask.metal').read_text().splitlines(True),src_text.splitlines(True),fromfile='a/src/metal/LocomotionTask.metal',tofile='b/'+name+'.metal')))
 subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
 cmd=json.load(open(A/(old+'-bindings.json')))['argv']
 for k,v in [('--metallib',str(lib)),('--state-trace',str(B/(name+'.tsv'))),('--crow-replay-pack',str(B/(name+'.json')))]:cmd[cmd.index(k)+1]=v
 (A/(name+'-bindings.json')).write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'launch-wing-on-plan.md'),'patch_sha256':h(A/(name+'.patch')),'shader_sha256':h(lib),'observer_sha256':h(C/'libmetalrobo.dylib'),'flight_source_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
 env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/(name+'.stdout')).write_text(p.stdout)
 with (A/(name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(name+' '+p.stderr[-3000:])
 raw=(B/(name+'.json')).read_bytes()
 with (A/(name+'.replay.json.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 print(name,'finished',flush=True)
