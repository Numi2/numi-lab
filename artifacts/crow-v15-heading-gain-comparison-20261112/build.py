from pathlib import Path
import hashlib,json,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-heading-gain-comparison-20261112";B=R/"build/crow-v15-heading-gain-comparison-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";F=R/"artifacts/crow-v15-flight-feedback-freeze-20261110";P=R/"build/crow-v15-continuous-launch-route-diagnostic-20261112"
subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(F),"-I",str(W/"include"),"-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(A/"flexed.metal"),"-o",str(B/"flexed.air")],check=True)
replaced={p.name:p for p in (W/"shaders").glob("*.air")};airs=[str((R/"build/crow-v15-walk-exact-bypass-20261112/World.air") if p.name=="MetalRoboWorld.air" else replaced.get(p.name,p)) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name not in ("MetalRoboLocomotionTask.air","MetalRoboFlappingWingDynamics.air")]
subprocess.run(["xcrun","-sdk","macosx","metallib",*airs,str(P/"wing.air"),str(B/"flexed.air"),"-o",str(B/"flexed.metallib")],check=True)
(A/"hashes.json").write_text(json.dumps({"source_sha256":hashlib.sha256((A/"flexed.metal").read_bytes()).hexdigest(),"metallib_sha256":hashlib.sha256((B/"flexed.metallib").read_bytes()).hexdigest()},indent=2)+"\n")
