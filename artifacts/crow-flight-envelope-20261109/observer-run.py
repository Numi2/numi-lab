from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-flight-envelope-20261109';B=R/'build/crow-flight-envelope-20261109'
lib=B/'observer-lib/libmetalrobo.dylib'
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
for name in ['yaw-negative','yaw-zero','yaw-positive']:
 shader=B/('observer-'+name+'.metallib')
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(B/(name+'.air')),str(B/'wrench-observer.air'),'-o',str(shader)],check=True)
 old=json.loads((A/(name+'-analysis.json')).read_text())
 entries={}
 for enabled in (False,True):
  flag='on' if enabled else 'off'
  cmd=old['argv'].copy()
  for arg,path in [('--metallib',shader),('--state-trace',B/('observer-'+name+'-'+flag+'-lane0.tsv'))]:
   cmd[cmd.index(arg)+1]=str(path)
  env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
  if enabled:env['CROW_WRENCH_TRACE']='1'
  else:env.pop('CROW_WRENCH_TRACE',None)
  p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
  if p.returncode:raise RuntimeError(p.stderr[-5000:])
  data=json.loads(p.stdout)
  if enabled:
   with (A/('observer-'+name+'.stderr.gz')).open('wb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(p.stderr.encode())
  entries[flag]={'argv':cmd,'first_episode':data['first_episode_navigation'],'failed_environment_steps':data['failed_environment_steps'],'trace_sha256':hashlib.sha256((B/('observer-'+name+'-'+flag+'-lane0.tsv')).read_bytes()).hexdigest()}
 report={'arm':name,'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'observer_on_off_trace_identical':entries['on']['trace_sha256']==entries['off']['trace_sha256'],'first_episode_identical':entries['on']['first_episode']==entries['off']['first_episode'],'runs':entries}
 (A/('observer-'+name+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(name,'trace_identical',report['observer_on_off_trace_identical'],'episodes_identical',report['first_episode_identical'],'failed_steps',entries['on']['failed_environment_steps'],entries['off']['failed_environment_steps'],flush=True)
