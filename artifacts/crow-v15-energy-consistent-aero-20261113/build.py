from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-energy-consistent-aero-20261113';B=R/'build/crow-v15-energy-consistent-aero-20261113';W=R/'build/crow-v15-wide-contact-isolation-20261111';old=(R/'build/crow-v15-wing-aero-alignment-20261110/FlappingWingDynamics.metal').read_text()
needle='''wing.unsteadyCoefficients.x * strokeFraction *
                    unsteadyDirection'''
replacement='''wing.unsteadyCoefficients.x * strokeFraction *
                    (unsteadyDirection - min(dot(unsteadyDirection, flow), 0.0f) * flow)'''
assert old.count(needle)==1;s=old.replace(needle,replacement);(A/'research-aero.metal').write_text(s);(A/'aero.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/v15-aero.metal',tofile='b/passive-research-aero.metal')))
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-c',str(A/'research-aero.metal'),'-o',str(B/'aero.air')],check=True)
replaced={p.name:p for p in (W/'shaders').glob('*.air')};base=R/'build/crow-v15-supported-wing-transition-20261113';airs=[str((base/'World.air') if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
for name in ('push-wing','hold-wing'):
 subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(B/'aero.air'),str(base/(name+'.air')),'-o',str(B/(name+'.metallib'))],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'build-hashes.json').write_text(json.dumps({str(p.relative_to(R)):h(p) for p in [A/'research-aero.metal',R/'build/crow-v15-wing-aero-alignment-20261110/FlappingWingDynamics.metal',B/'aero.air',B/'push-wing.metallib',B/'hold-wing.metallib']},indent=2)+'\n');print((A/'build-hashes.json').read_text())
old='crow-v15-aero-power-audit-20261113';new='crow-v15-energy-consistent-aero-20261113';s=(R/'artifacts'/old/'run.py').read_text().replace(old,new).replace("C=R/'build/crow-v15-supported-wing-transition-20261113'", "C=R/'build/crow-v15-energy-consistent-aero-20261113'").replace("L=R/'build/crow-v15-energy-consistent-aero-20261113'", "L=R/'build/crow-v15-aero-power-audit-20261113'");(A/'run.py').write_text(s)
