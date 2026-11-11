from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-dagger-isolated-labels-control-20261111';B=R/'build/crow-v15-dagger-isolated-labels-control-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/"artifacts/crow-v15-teacher-launch-parity-candidate-20261111/parity.metal").read_text()
a = '            // Policy observations do not expose privileged teacher commands.\n'
b = '            state.threatTeacher=float4(yawFrameLinear.y,roll,baseAngular.x,landingPhase);\n'
c = '    else if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u) {'
assert base.count(a)==1 and base.count(b)==1 and base.count(c)==1
modified=base.replace(a, '            if (dispatch.assistance.x < 0.5f) {\n'+a).replace(b,b+'            }\n').replace(c, '    if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && !(dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f && float(state.episode.x)*dispatch.timing.x >= 8.0f)) {')

variants={"parity":modified}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
for name,s in variants.items():
 source=A/(name+'.metal');source.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(source),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);out[name]={'source_sha256':h(source),'metallib_sha256':h(ml)}
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
