from pathlib import Path
import json,subprocess,hashlib,time
d=Path(__file__).resolve().parent;p=d.parent;w=Path('/Users/numi/Work/skin-fat-native-integration-209')
old=p/'sparse-mass-transfer-667/native-overlay-001'
build=d/'publication-native-build-001';build.mkdir(exist_ok=True)
cmd=json.loads((old/'compile-result.json').read_text())['compiler_argv']
cmd[cmd.index('-ivfsoverlay')+1]=str(d/'publication-overlay.json')
cmd[cmd.index('-o')+1]=str(build/'probe.o')
cmd[1:1]=['-I'+str(d/'publication-source/apps'),'-I'+str(w/'apps')]
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True)
(build/'compile.stdout').write_text(r.stdout);(build/'compile.stderr').write_text(r.stderr)
rec={'compiler_argv':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-t}
if r.returncode==0:
 rec['object_sha256']=hashlib.sha256((build/'probe.o').read_bytes()).hexdigest()
 rec['include_pins']={name:str(d/'publication-source/apps'/name) in r.stderr for name in ['NumiHumanPhysicalSkinChildMassMapping.hpp','NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp']}
(build/'compile-result.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),flush=True)
if r.returncode:print(r.stderr[-5000:]);raise SystemExit(r.returncode)
link=json.loads((old/'link-002-result.json').read_text())['argv']
link[link.index('-o')+1]=str(build/'numi-human-native')
link[link.index(str(old/'numilab_human_myosim_visual_probe.mm.o'))]=str(build/'probe.o')
t=time.monotonic();r=subprocess.run(link,cwd=w/'build-native',capture_output=True,text=True)
(build/'link.stdout').write_text(r.stdout);(build/'link.stderr').write_text(r.stderr)
rec={'argv':link,'exit_code':r.returncode,'seconds':time.monotonic()-t,'run_executed':False}
if r.returncode==0:rec['binary_sha256']=hashlib.sha256((build/'numi-human-native').read_bytes()).hexdigest()
(build/'link-result.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),r.stderr[-3000:],flush=True)
