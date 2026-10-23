from pathlib import Path
import shutil,subprocess,json,time,hashlib
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D=E/"sparse-contact-arithmetic-138"; D.mkdir(exist_ok=False)
O=D/"source-overlay"; shutil.copytree(E/"sparse-bone-contact-135/source-overlay",O,ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2("/tmp/numi-sparse-contact135-arithmetic-001/source-overlay/matter/src/metal/numi_human.metalinc",O/"matter/src/metal/numi_human.metalinc")
B=D/"build"; B.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
st=[]
for name,args in [("metal",["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-fno-fast-math","-I",str(O/"matter/src"),"-I",str(R/"matter/src"),"-I",str(R/"matter/include"),"-I",str(R/"include"),"-c",str(O/"matter/src/matter_kernels.metal"),"-o",str(B/"candidate.air")]),("metallib",["xcrun","-sdk","macosx","metallib",str(B/"candidate.air"),"-o",str(B/"candidate.metallib")])]:
 t=time.monotonic();p=subprocess.run(args,cwd=R,capture_output=True,text=True);(B/(name+".log")).write_text(p.stdout+p.stderr);st.append(dict(name=name,argv=args,returncode=p.returncode,wall_seconds=time.monotonic()-t));assert p.returncode==0
p=E/"sparse-bone-contact-135/build-native-002/libmetalrobo.dylib";shutil.copy2(p,B/p.name)
r=dict(stages=st,source_sha256=sha(O/"matter/src/metal/numi_human.metalinc"),metallib_sha256=sha(B/"candidate.metallib"),runtime_sha256=sha(B/p.name),runtime_reused_from=str(p))
(B/"build-manifest.json").write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
