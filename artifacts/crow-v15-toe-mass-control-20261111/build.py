from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-toe-mass-control-20261111";B=R/"build/crow-v15-toe-mass-control-20261111";W=R/"build/crow-v15-wide-contact-isolation-20261111";J=R/"build/crow-v15-hip-roll-journey-integration-20261111"
base=(J/"c_api.cpp").read_text();needle="constexpr float mass=0.00005f, hx=0.009f, hy=0.0025f, hz=0.0092f;";assert base.count(needle)==1
s=base.replace(needle,needle.replace("0.00005f","0.00010f"));(B/"c_api.cpp").write_text(s);(A/"host.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/reference.cpp",tofile="b/toe-thickness.cpp")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();compile=next(x for x in lines if " -c "+str(R/"src/c_api.cpp") in x);link=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(compile);obj=B/"c_api.cpp.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"c_api.cpp");args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"artifacts/crow-v15-two-segment-frozen-flight-setup-20261111"),"-I"+str(R/"artifacts/crow-clearance-observer-20261109")];subprocess.run(args,cwd=R/"build",check=True);args=shlex.split(link[len(": && "):-len(" && :")]);lib=B/"libmetalrobo.dylib";args[args.index("-o")+1]=str(lib);args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(obj)
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 old="CMakeFiles/metalrobo.dir/"+n+".o";assert old in args;args[args.index(old)]=str(J/"MetalWorld.mm.o" if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"host_source_sha256":h(B/"c_api.cpp"),"host_sha256":h(lib),"toe_mass_kg":.00010,"scope":"experimental constant damping; no biological calibration"},indent=2)+"\n")
