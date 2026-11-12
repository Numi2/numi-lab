from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-gait-slip-observer-20261112';B=R/'build/crow-v15-gait-slip-observer-20261112';J=R/'build/crow-v15-hip-roll-journey-integration-20261111';W=R/'build/crow-v15-wide-contact-isolation-20261111'
cmd=['xcrun','clang++','-std=c++20','-O2','-I'+str(W/'include'),'-Iinclude',str(A/'kinematics.cpp'),'-L'+str(J),'-lmetalrobo','-Wl,-rpath,'+str(J),'-o',str(B/'kinematics')]
subprocess.run(cmd,check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'kinematics-inputs.json').write_text(json.dumps({'compile':cmd,'source_sha256':h(A/'kinematics.cpp'),'binary_sha256':h(B/'kinematics'),'library_sha256':h(J/'libmetalrobo.dylib')},indent=2)+'\n')
