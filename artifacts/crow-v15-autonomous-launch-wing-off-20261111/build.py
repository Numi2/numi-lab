from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-autonomous-launch-wing-off-20261111';B=R/'build/crow-v15-autonomous-launch-wing-off-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/'artifacts/crow-v15-aligned-contact-release-control-20261111/release.metal').read_text()
modified=(A/'release.metal').read_text()
variants={'release':modified}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
for name,s in variants.items():
 source=A/(name+'.metal');source.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(source),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);out[name]={'source_sha256':h(source),'metallib_sha256':h(ml)}
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
