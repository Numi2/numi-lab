from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perch-transient-grip-control-20261111';B=R/'build/crow-v15-perch-transient-grip-control-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111';F=R/'artifacts/crow-v15-flight-feedback-freeze-20261110';base=(R/"artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/balanced.metal").read_text()
needle='''                const uint ramp=loaded[side] ? min(oldRamp+1u,34u) : (oldRamp>0u ? oldRamp-1u : 0u);'''
replacement='''                // Isolated contact trial: retain a transient toe closure for
                // at most 0.2 s after a physically measured perch impulse.
                const uint shift=23u+4u*side;
                const uint oldHold=(state.status.w>>shift)&15u;
                const uint hold=loaded[side] ? 10u : (oldHold>0u ? oldHold-1u : 0u);
                state.status.w=(state.status.w & ~(15u<<shift)) | (hold<<shift);
                const uint ramp=(loaded[side] || hold>0u) ? min(oldRamp+1u,34u) : (oldRamp>0u ? oldRamp-1u : 0u);'''
assert base.count(needle)==1
variants={"hold":base.replace(needle,replacement)}

h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
for name,s in variants.items():
 source=A/(name+'.metal');source.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/reference.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(F),'-I',str(W/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(source),'-o',str(air)],check=True);replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[str(replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')];ml=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(air),'-o',str(ml)],check=True);out[name]={'source_sha256':h(source),'metallib_sha256':h(ml)}
(A/'hashes.json').write_text(json.dumps(out,indent=2)+'\n')
