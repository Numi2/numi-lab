from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-passive-yaw-hypothesis-screen-20261112";B=R/"build/crow-v15-passive-yaw-hypothesis-screen-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-c",str(A/"aero.metal"),"-o",str(B/"aero.air")],check=True)
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"bank.metal"),"-o",str(B/"bank.air")],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str(replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
for label,aero in (("control",R/"build/crow-v15-wing-aero-alignment-20261110/aero.air"),("passive",B/"aero.air")):
 subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(aero),str(B/"bank.air"),"-o",str(B/(label+".metallib"))],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({x:h(B/(x+".metallib")) for x in ("control","passive")},indent=2)+"\n")
