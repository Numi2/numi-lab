from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-standing-restitution-control-20261112';B=R/'build/crow-v15-standing-restitution-control-20261112';W=R/'build/crow-v15-wide-contact-isolation-20261111';T=R/'build/crow-v15-rear-toe-matched-trial-20261112/control.air';orig=(R/'src/metal/MetalWorldContact.metal').read_text();zero=orig
x='-sourceCone.restitution * incoming';assert zero.count(x)==1;zero=zero.replace(x,'0.0f * incoming')
x='-contact.response.x *\n                            witness';assert zero.count(x)==1;zero=zero.replace(x,'-0.0f *\n                            witness')
(A/'zero-restitution.patch').write_text(''.join(difflib.unified_diff(orig.splitlines(True),zero.splitlines(True),fromfile='a/MetalWorldContact.metal',tofile='b/MetalWorldContact.metal')))
outputs={}
boost=orig
x='-sourceCone.restitution * incoming'; assert boost.count(x)==1; boost=boost.replace(x,'-1.0f * incoming')
x='-sourceCone.restitutionThreshold'; assert boost.count(x)==1; boost=boost.replace(x,'-0.0f')
x='-contact.response.x *\n                            witness'; assert boost.count(x)==1; boost=boost.replace(x,'-1.0f *\n                            witness')
x='-evaluatedCones[\n                             constraintBase +\n                             localConstraint\n                         ].restitutionThreshold'; assert boost.count(x)==1; boost=boost.replace(x,'-0.0f')
for name,src in [('boost',boost)]:
 source=A/(name+'.metal');source.write_text(src);air=B/(name+'.air');lib=B/(name+'.metallib');cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-c',str(source),'-o',str(air)];subprocess.run(cmd,check=True)
 replaced={p.name:p for p in (W/'shaders').glob('*.air')};airs=[]
 for p in sorted((R/'build/shaders').glob('MetalRobo*.air')):
  if p.name in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'):continue
  airs.append(str(air if p.name=='MetalRoboWorldContact.air' else R/'build/crow-v15-walk-exact-bypass-20261112/World.air' if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)))
 subprocess.run(['xcrun','-sdk','macosx','metallib',*airs,str(R/'build/crow-v15-wing-aero-alignment-20261110/aero.air'),str(T),'-o',str(lib)],check=True)
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();outputs[name]={'source_sha256':h(source),'air_sha256':h(air),'metallib_sha256':h(lib),'task_air_sha256':h(T)}
(A/'boost-hashes.json').write_text(json.dumps(outputs,indent=2)+'\n')
