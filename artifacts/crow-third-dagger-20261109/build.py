from pathlib import Path
import difflib,hashlib,json,os,subprocess
R=Path.cwd();schedule=os.environ['CROW_SCHEDULE'];base_step,delta={'early':(1000,50),'nearcritical':(1050,25),'critical':(1100,25),'late':(1300,100),'recovery':(1600,50),'terminal':(2100,100)}[schedule]
A=R/'artifacts/crow-third-dagger-20261109'/schedule;B=R/'build/crow-third-dagger-20261109'/schedule;A.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
base=(R/'build/crow-teacher-late-recovery-20261109/teacher.metal').read_text()
old='const bool privilegedCommand = (dispatch.sampling.w != 0u && dispatch.assistance.x < 0.5f);'
new=f'const bool privilegedCommand = dispatch.sampling.w != 0u && (dispatch.assistance.x < 0.5f || state.episode.x >= ({base_step}u + {delta}u*(environment % 4u)));'
assert base.count(old)==1
s=base.replace(old,new);src=B/'recovery.metal';src.write_text(s)
(A/'recovery.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/development/late-recovery-teacher.metal',tofile='b/development/recovery.metal')))
air=B/'recovery.air';shader=B/'recovery.metallib'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name!='MetalRoboLocomotionTask.air')
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(air),'-o',str(shader)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'shader.json').write_text(json.dumps({'schedule':schedule,'source_sha256':sha(src),'shader_sha256':sha(shader),'frozen_flight_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'scope':'diagnostic teacher takeover; teacher-disabled student behavior must match production'},indent=2)+'\n')
