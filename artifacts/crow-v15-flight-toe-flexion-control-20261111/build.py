from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-flight-toe-flexion-control-20261111';B=R/'build/crow-v15-flight-toe-flexion-control-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/"artifacts/crow-v15-hip-roll-walk-launch-controls-20261111/gait-080.metal").read_text()
flight=(R/"artifacts/crow-v15-journey-bounded-flight-screen-20261111/tracking.metal").read_text()
a="const float proximal=toe==3u ? min(.04f*float(ramp),1.15f) : min(.03f*float(frontRamp),1.05f);"
b="state.status.w=(previous & ~(4096u|8192u|16384u|32768u)) | low1 | low2 | retract | (released?8192u:0u);"
def update(src):
 assert src.count(a)==1 and src.count(b)==1
 src=src.replace(a,"const float perchTarget=toe==3u ? min(.04f*float(ramp),1.15f) : min(.03f*float(frontRamp),1.05f); const float flightTarget=.025f*float((state.status.w>>23u)&31u); const float proximal=max(perchTarget,flightTarget);")
 src=src.replace(b,b+"\n            const uint oldToeFlight=(state.status.w>>23u)&31u;\n            const bool curlFlight=released && seconds>=7.4f && state.navigation.z<4.0f;\n            const uint nextToeFlight=curlFlight ? min(oldToeFlight+1u,28u) : (oldToeFlight>0u ? oldToeFlight-1u : 0u);\n            state.status.w=(state.status.w & ~(31u<<23u)) | (nextToeFlight<<23u);")
 return src
variants={"curl-gait":update(base),"curl-flight":update(flight)}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
for name,s in variants.items():
 source=A/(name+'.metal');source.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(source),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);out[name]={'source_sha256':h(source),'metallib_sha256':h(ml)}
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
