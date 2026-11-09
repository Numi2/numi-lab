from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-release-timing-20261109';B=R/'build/crow-v15-release-timing-20261109';base=(R/'build/crow-v15-toe-reactive-20261109/active.metal').read_text();assert base.count("55.0f")==2
for name,seconds in [("positive","45.0f"),("negative","46.0f")]:
 s=base.replace("55.0f",seconds);src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/toe-active.metal",tofile="b/release-"+name+".metal")))
 air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
