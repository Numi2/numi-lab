from pathlib import Path
import hashlib,subprocess
R=Path.cwd();B=R/"build/crow-v15-gait-coordinated-confirmation-20261110"
for arm in ("push-trigger","control-trigger"):
 air=B/(arm+".air");lib=B/(arm+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(B/(arm+".metal")),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(arm,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
