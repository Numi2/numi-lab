from pathlib import Path
import subprocess,os,json,gzip,hashlib,shutil,difflib
R=Path.cwd();A=R/'artifacts/crow-v15-corrected-wing-control-sweep-20261110';B=R/'build/crow-v15-corrected-wing-control-sweep-20261110';reference=(R/'src/metal/CrowFlightControllerV1.metal').read_text(); tracking=(R/'build/crow-v15-toe-flight-screen-20261110/tracking.metal').read_text();base=json.loads((R/'artifacts/crow-v15-wing-aero-alignment-20261110/flight-analysis.json').read_text())['argv'];host=R/'build/crow-v15-wing-inertia-prototype-20261110';
needle='const float speedBias = clamp(0.55f + 1.40f * (state.commandExtension.z - max(state.commandAndPhase.x, 0.15f)), 0.25f, 0.95f);';assert reference.count(needle)==1
for label,bias in [('zero','0.0f'),('positive','0.25f'),('negative','-0.25f')]:
 D=B/label;D.mkdir(exist_ok=True); H=reference.replace(needle,'const float speedBias = '+bias+';');(D/'CrowFlightControllerExperiment.metal').write_text(H);(A/(label+'-controller.patch')).write_text(''.join(difflib.unified_diff(reference.splitlines(True),H.splitlines(True),fromfile='a/CrowFlightControllerV1.metal',tofile='b/experimental-'+label+'.metal')))
 t=tracking.replace('#include "CrowFlightControllerV1.metal"','#include "CrowFlightControllerExperiment.metal"');assert t!=tracking;(D/'tracking.metal').write_text(t)
 cmds=[['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(D),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(D/'tracking.metal'),'-o',str(D/'tracking.air')]]
 airs=[str(x) for x in (R/'build/shaders').glob('MetalRobo*.air') if x.name not in ('MetalRoboFlappingWingDynamics.air','MetalRoboLocomotionTask.air')];airs += [str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(D/'tracking.air')];cmds += [['xcrun','-sdk','macosx','metallib',*airs,'-o',str(D/'tracking.metallib')]]
 with (A/(label+'-build.log')).open('w') as f:
  for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
 cmd=base[:]
 for k,v in [('--metallib',str(D/'tracking.metallib')),('--steps','900'),('--state-trace',str(D/'flight.tsv'))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(host),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/(label+'.stdout')).write_text(p.stdout)
 with gzip.GzipFile(filename=str(A/(label+'.stderr.gz')),mode='wb',mtime=0) as f:f.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(label+' '+p.stderr[-2000:])
 d=json.loads(p.stdout);e=d['first_episode_navigation'];report={'label':label,'pronation_bias':float(bias[:-1]),'metallib_sha256':hashlib.sha256((D/'tracking.metallib').read_bytes()).hexdigest(),'host_sha256':hashlib.sha256((host/'libmetalrobo.dylib').read_bytes()).hexdigest(),'command':cmd,'failed_environment_steps':d['failed_environment_steps'],'first_episode':e,'scope':'Exploratory 900-step short screen with estimated wing and specialized force geometry; no qualification.'};(A/(label+'-analysis.json')).write_text(json.dumps(report,indent=2)+'\n')
 with (D/'flight.tsv').open('rb') as f,gzip.GzipFile(filename=str(A/(label+'.tsv.gz')),mode='wb',mtime=0) as g:shutil.copyfileobj(f,g)
 print(label,'first episode steps',e['steps_by_environment'],'physical failures',e['physical_failure_count'],'solver failures',d['failed_environment_steps'],flush=True)
