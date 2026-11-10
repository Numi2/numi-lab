from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/"artifacts/crow-v15-gait-coordinated-stance-20261110";B=R/"build/crow-v15-gait-coordinated-stance-20261110";base=(R/"build/crow-v15-sustained-gait-20261110/push-trigger.metal").read_text()
for name,period,power in [("coord07520","0.75",1),("coord07535","0.75",1),("coord10020","1.00",1),("coord10035","1.00",1)]:
 s=base.replace("sin(kTwoPi*seconds/0.50f)*(leg==0u ? 1.0f : -1.0f)","sin(kTwoPi*seconds/"+period+"f)*(leg==0u ? 1.0f : -1.0f)")
 assert s!=base
 if power==2:s=s.replace("const float lift=max(swing,0.0f);","const float lift=pow(max(swing,0.0f),2.0f);").replace("(swing>=0.0f ? -0.30f*swing : -0.042f*swing)","(swing>=0.0f ? -0.30f*lift : -0.042f*swing)")
 k="0.20" if name.endswith("20") else "0.35"
 before="navigationTeacherRequested=joint==0u ? (swing>=0.0f ? -0.30f*swing : -0.042f*swing) : joint==1u ? 0.40f*lift : -0.15f*lift;"
 after="navigationTeacherRequested=joint==0u ? (swing>=0.0f ? -0.30f*swing : -0.042f*swing + 0.536f*"+k+"f*max(-swing,0.0f)) : joint==1u ? 0.40f*lift - "+k+"f*max(-swing,0.0f) : -0.15f*lift + 0.577f*"+k+"f*max(-swing,0.0f);"
 assert s.count(before)==1
 s=s.replace(before,after)
 src=B/(name+".metal");src.write_text(s);(A/(name+".patch")).write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/sustained-push.metal",tofile="b/"+name+".metal")));air=B/(name+".air");lib=B/(name+".metallib");subprocess.run(["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(R/"include"),"-I",str(R/"src/metal"),"-c",str(src),"-o",str(air)],check=True);subprocess.run(["xcrun","-sdk","macosx","metallib",*[str(p) for p in sorted((R/"build/shaders").glob("MetalRobo*.air")) if p.name!="MetalRoboLocomotionTask.air"],str(air),"-o",str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
