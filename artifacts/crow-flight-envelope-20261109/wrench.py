from pathlib import Path
import difflib,gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-flight-envelope-20261109';B=R/'build/crow-flight-envelope-20261109'
lib=B/'wrench-lib/libmetalrobo.dylib'
source=(R/'src/metal/FlappingWingDynamics.metal').read_text()
def one(s,a,b):
 if s.count(a)!=1:raise RuntimeError('source mismatch: '+a[:80])
 return s.replace(a,b)
s=one(source,'''    constant const MRAeroFuselageGPU& fuselage [[buffer(6)]],
    const uint environment''','''    constant const MRAeroFuselageGPU& fuselage [[buffer(6)]],
    device float* diagnosticTeacherActions [[buffer(7)]],
    const uint environment''')
s=one(s,'''    float wingStrokeSpeedSquared = 0.0f;''','''    const uint debugRoot = environment * dispatch.bodyStride + dispatch.rootBodyIndex;
    const float debugBeforeAeroYaw = wrenches[debugRoot].torque.z;
    float debugWingYawAboutRoot = 0.0f;
    float wingStrokeSpeedSquared = 0.0f;''')
s=one(s,'''        const float3 wingTorque = torque - cross(rootToWingCOM, force);''','''        debugWingYawAboutRoot += torque.z;
        const float3 wingTorque = torque - cross(rootToWingCOM, force);''')
needle='''        wrench.force.xyz += tailForce;
        wrenches[wrenchIndex] = wrench;
    }
}'''
s=one(s,needle,'''        wrench.force.xyz += tailForce;
        wrenches[wrenchIndex] = wrench;
    }
    // Measurement only: chunk=1, v15 has exactly 18 task actions.
    // The final slot's tail moment uses the aero reference point, not the
    // actual tail COM, and is therefore an approximation.
    diagnosticTeacherActions[environment * 18u + 15u] = debugBeforeAeroYaw;
    diagnosticTeacherActions[environment * 18u + 16u] = debugWingYawAboutRoot;
    diagnosticTeacherActions[environment * 18u + 17u] =
        wrenches[debugRoot].torque.z - debugBeforeAeroYaw +
        cross(rootToTail, tailForce).z;
}''')
(A/'wrench-shader.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='a/src/metal/FlappingWingDynamics.metal',tofile='b/src/metal/FlappingWingDynamics.metal')))
src=B/'wrench-aero.metal';src.write_text(s);air=B/'wrench-aero.air'
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
other=sorted(p for p in (R/'build/shaders').glob('MetalRobo*.air') if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air'))
for name in ['yaw-negative','yaw-zero','yaw-positive']:
 shader=B/('wrench-'+name+'.metallib')
 subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in other],str(B/(name+'.air')),str(air),'-o',str(shader)],check=True)
 old=json.loads((A/(name+'-analysis.json')).read_text());cmd=old['argv'].copy()
 for arg,path in [('--metallib',shader),('--state-trace',B/('wrench-'+name+'-lane0.tsv'))]:
  cmd[cmd.index(arg)+1]=str(path)
 env={**os.environ,'CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
 run=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
 (A/('wrench-'+name+'.stdout')).write_text(run.stdout)
 with (A/('wrench-'+name+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(run.stderr.encode())
 if run.returncode:raise RuntimeError(run.stderr[-5000:])
 data=json.loads(run.stdout)
 trace=(B/('wrench-'+name+'-lane0.tsv')).read_bytes();original=(B/(name+'-lane0.tsv')).read_bytes()
 report={'arm':name,'argv':cmd,'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'trace_matches_uninstrumented':trace==original,'first_episode_matches_uninstrumented':data['first_episode_navigation']==old['first_episode'],'failed_environment_steps':data['failed_environment_steps']}
 (A/('wrench-'+name+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(name,report,flush=True)
