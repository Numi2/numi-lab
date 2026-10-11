import json,subprocess,time,hashlib
from pathlib import Path
W=Path('/Users/n/numi-lab-parent-bindings-1353')
B=Path('/Users/n/numi-lab-parent-bindings-build-1353-004');B.mkdir(exist_ok=False);(B/'bin').mkdir()
old=json.loads(Path('/Users/n/numi-human-retained-delivery-20261009/viewer-footer-1256/build-001/build-report.json').read_text())
sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
commands=[]
for entry in old['commands']:
 argv=entry['argv']
 out=[]
 for a in argv:
  a=a.replace('/Users/n/numi-human-viewer-footer-build-1256-001',str(B)).replace('/Users/n/numi-human-viewer-footer-1256',str(W))
  a=a.replace('/Users/n/numi-human-contoured-bed-build-1196','/Users/n/numi-lab-complete-body-capture-build-1276-001')
  a=a.replace('/Applications/Xcode-26.6.0.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk',sdk)
  out.append(a)
 commands.append(dict(phase=entry['phase'],argv=out))
inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'apps/numilab_human_myosim_visual_probe.mm',W/'apps/NumiHumanRestingVisual.hpp',W/'apps/NumiHumanRestingTissueParents.hpp']}
report=dict(source=str(W),source_head=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),commands=[],input_sha256=inputs,native_execution=False)
for c in commands:
 at=time.monotonic()
 with (B/(c['phase']+'.log')).open('w') as f:r=subprocess.run(c['argv'],stdout=f,stderr=subprocess.STDOUT)
 c.update(returncode=r.returncode,wall_seconds=time.monotonic()-at);report['commands'].append(c)
 (B/'build-report.json').write_text(json.dumps(report,indent=2))
 print(c['phase'],r.returncode,flush=True)
 if r.returncode:break
report['inputs_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in inputs.items())
(B/'build-report.json').write_text(json.dumps(report,indent=2))
