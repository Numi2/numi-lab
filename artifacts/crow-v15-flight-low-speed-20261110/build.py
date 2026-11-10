from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-flight-low-speed-20261110";B=R/"build/crow-v15-flight-low-speed-20261110";base=(R/"build/crow-v15-toe-flight-screen-20261110/tracking.metal").read_text();old="state.commandAndPhase.xyz = float3(0.70f, 0.0f, rate);";assert base.count(old)==1
for arm,val in (("slow","0.25f"),("hover","0.00f"),("reverse","-0.15f")):
 s=base.replace(old,"state.commandAndPhase.xyz = float3("+val+", 0.0f, rate);");p=B/(arm+".metal");p.write_text(s);(A/(arm+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/tracking.metal",tofile="b/"+arm+".metal")))
 air=B/(arm+".air");lib=B/(arm+".metallib")
 subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(p),"-o",str(air)],check=True)
 subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True)
 print(arm,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
