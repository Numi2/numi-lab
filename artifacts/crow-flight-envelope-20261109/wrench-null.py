from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-flight-envelope-20261109';B=R/'build/crow-flight-envelope-20261109'
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
lib=B/'wrench-lib/libmetalrobo.dylib'
for name in ['yaw-negative','yaw-zero','yaw-positive']:
 shader=B/('wrench-null-'+name+'.metallib')
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(B/(name+'.air')),'-o',str(shader)],check=True)
 prev=json.loads((A/('wrench-'+name+'-verification.json')).read_text())
 cmd=prev['argv'].copy()
 for arg,path in [('--metallib',shader),('--state-trace',B/('wrench-null-'+name+'-lane0.tsv'))]:
  cmd[cmd.index(arg)+1]=str(path)
 env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180,check=True)
 data=json.loads(p.stdout)
 trace=(B/('wrench-null-'+name+'-lane0.tsv')).read_bytes()
 instrumented=(B/('wrench-'+name+'-lane0.tsv')).read_bytes()
 expected=json.loads((A/('wrench-'+name+'.stdout')).read_text())
 result={'arm':name,'argv':cmd,'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'selected_trace_identical_to_instrumented':trace==instrumented,'first_episode_identical_to_instrumented':data['first_episode_navigation']==expected['first_episode_navigation'],'failed_environment_steps':data['failed_environment_steps']}
 (A/('wrench-null-'+name+'-verification.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(name,result['selected_trace_identical_to_instrumented'],result['first_episode_identical_to_instrumented'],result['failed_environment_steps'],flush=True)
