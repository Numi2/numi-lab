from pathlib import Path
import hashlib,json,subprocess,datetime
BASE=Path(__file__).parent
OLD=Path('/Users/n/numi-human-retained-delivery-20261009/accepted-force-observer-1253/paired-smoke-001')
SOURCE=Path('/Users/n/numi-human-viewer-footer-1256')
BUILD=Path('/Users/n/numi-human-viewer-footer-build-1256-001')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
old=OLD/'window/run-declaration.json'
assert sha(old)=='46408bf228d4b80a0dfd54880c4885e058dad3513e6ce683cddf487143bb2a46'
d=json.loads(old.read_text())
out=BASE/'window/native-run';out.parent.mkdir()
oldout=str(OLD/'window/native-run')
oldbuild='/Users/n/numi-human-accepted-force-build-1253-002'
oldsource='/Users/n/numi-human-accepted-force-observer-1253'
d['argv']=[x.replace(oldout,str(out)).replace(oldbuild,str(BUILD)) for x in d['argv']]
d['argv']=[('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=512' if x.startswith('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=') else x) for x in d['argv']]
d['owner_cli_preview']['native_argv']=[x.replace(oldout,str(out)).replace(oldbuild,str(BUILD)) for x in d['owner_cli_preview']['native_argv']]
e=d['owner_cli_preview']['recorded_invocation_environment']
d['owner_cli_preview']['recorded_invocation_environment']={k:(v.replace(oldout,str(out)).replace(oldbuild,str(BUILD)) if isinstance(v,str) else v) for k,v in e.items()}
d['owner_cli_preview']['recorded_invocation_environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='512'
assets={}
for path,h in d['immutable_assets'].items():
 p=Path(path)
 if path.startswith(str(OLD)) or path.startswith(str(OLD.parent/'build-002')):continue
 p=Path(path.replace(oldsource,str(SOURCE)).replace(oldbuild,str(BUILD)))
 assets[str(p)]=sha(p)
for p in [Path(__file__),BASE/'launch_arm.py',old,BASE.parent/'build-001/build-report.json',BASE.parent/'build-001/build.py',SOURCE/'apps/NumiHumanRestingWindow.hpp',SOURCE/'apps/NumiHumanRestingVisual.hpp',BUILD/'bin/numi-human-native']:
 assets[str(p)]=sha(p)
d['immutable_assets']=dict(sorted(assets.items()))
d['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
d['capture_steps']=[512]
d['launch_adapter']={'path':str(BASE/'launch_arm.py'),'sha256':sha(BASE/'launch_arm.py')}
d['source_identity']={'path':str(SOURCE),'revision':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).splitlines()}
d['human_identity']['revision']=subprocess.check_output(['git','-C',d['human_identity']['path'],'rev-parse','HEAD'],text=True).strip()
d['human_identity']['status']=subprocess.check_output(['git','-C',d['human_identity']['path'],'status','--porcelain'],text=True).splitlines()
d['physical_scope']={'same_current_FHL_scene_and_physical_control_settings_as_force_window':True,'changes':'AppKit measurement layout and explicit footer line break; omit redundant step0 export','native_runtime_library_sha256':sha(BUILD/'lib/libmetalrobo.dylib')}
d['qualification_boundary']='1.024 s presentation smoke and unchanged accepted traces only; not long-horizon or anatomy qualification'
write(out.parent/'run-declaration.json',d)
guard={'schema':'numi.human.native-smoke-launch-guard.v1','launcher_path':str(BASE/'launch_arm.py'),'launcher_sha256':sha(BASE/'launch_arm.py'),'arms':{'window':{'run_declaration':str(out.parent/'run-declaration.json'),'run_declaration_sha256':sha(out.parent/'run-declaration.json'),'native_output':str(out)}}}
write(BASE/'launch-guard.json',guard)
print(json.dumps(guard,indent=2))
