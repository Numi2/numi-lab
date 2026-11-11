from pathlib import Path
import difflib, hashlib, json, shlex, subprocess
R=Path.cwd(); A=R/"artifacts/crow-v15-knee-compliance-screen-20261111"; B=R/"build/crow-v15-knee-compliance-screen-20261111"; W=R/"build/crow-v15-wide-contact-isolation-20261111"; P=R/"build/crow-v15-landing-phase-observer-20261111"; H=R/"build/crow-v15-hip-roll-journey-integration-20261111"
base=(H/"c_api.cpp").read_text(); needle=" auto oldBodies=m.bodies;auto oldNames=m.bodyNames;auto oldJoints=m.joints;"; assert base.count(needle)==1
replacement=""" unsigned kneeCount=0;
 for (std::size_t i=0; i<m.dofNames.size(); ++i) {
   if (m.dofNames[i] == "crow_left_knee_pitch" || m.dofNames[i] == "crow_right_knee_pitch") {
     m.dofs[i].drive.x *= 0.4f; m.dofs[i].drive.y *= std::sqrt(0.4f); ++kneeCount;
   }
 }
 crowToeNeed(kneeCount==2u,"expected two knee drives");
"""+needle
s=base.replace(needle,replacement); (B/"c_api.cpp").write_text(s); (A/"host.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="a/reference.cpp",tofile="b/knee-compliance.cpp")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines(); compile=next(x for x in lines if " -c "+str(R/"src/c_api.cpp") in x); link=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x); args=shlex.split(compile); obj=B/"c_api.cpp.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]: args[args.index(k)+1]=v
args[-1]=str(B/"c_api.cpp"); args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"artifacts/crow-v15-two-segment-frozen-flight-setup-20261111"),"-I"+str(R/"artifacts/crow-clearance-observer-20261109")]; subprocess.run(args,cwd=R/"build",check=True)
args=shlex.split(link[len(": && "):-len(" && :")]); lib=B/"libmetalrobo.dylib"; args[args.index("-o")+1]=str(lib); args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(obj)
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 old="CMakeFiles/metalrobo.dir/"+n+".o"; assert old in args; args[args.index(old)]=str(P/"MetalWorld.mm.o" if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); (A/"build-hashes.json").write_text(json.dumps({"source_sha256":h(B/"c_api.cpp"),"host_sha256":h(lib),"reference_host_sha256":h(P/"libmetalrobo.dylib")},indent=2)+"\n")
print("isolated host built")
