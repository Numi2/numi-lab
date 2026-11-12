from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-uniform-height-no-step-20261112";B=R/"build/crow-v15-uniform-height-no-step-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110"
for n in ("flexed","wing"):
 subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/(n+".metal")),"-o",str(B/(n+".air"))],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")}
airs=[str((R/"build/crow-v15-walk-exact-bypass-20261112/World.air") if p.name=="MetalRoboWorld.air" else replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(B/"wing.air"),str(B/"flexed.air"),"-o",str(B/"flexed.metallib")],check=True)
(A/"hashes.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (A/"flexed.metal",A/"wing.metal",B/"flexed.metallib")},indent=2)+"\n")
