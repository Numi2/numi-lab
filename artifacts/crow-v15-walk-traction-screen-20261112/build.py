from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-walk-traction-screen-20261112";B=R/"build/crow-v15-walk-traction-screen-20261112";W=R/"build/crow-v15-wide-contact-isolation-20261111";H=R/"build/crow-v15-substep-target-interpolation-screen-20261112";S=R/"build/crow-v15-gait-slip-observer-20261112"
old=(S/"c_api.cpp").read_text();needle=" std::string why;crowToeNeed(m.valid(&why),\"model invalid: \"+why);return p;";assert old.count(needle)==1
insert=""" if (std::getenv("CROW_EXPERIMENT_TRACTION_HIGH") != nullptr) {
  auto material=m.materials.at(0);material.friction.x=1.8f;material.friction.y=1.35f;
  const unsigned materialIndex=m.materials.size();m.materials.push_back(material);m.world.materialCount=m.materials.size();unsigned changed=0;
  for(auto& shape:m.shapes) {const auto& name=m.bodyNames.at(shape.bodyIndex);if(name=="crow_left_foot" || name=="crow_right_foot" || name.find("_toe_")!=std::string::npos){shape.materialIndex=materialIndex;++changed;}}
  crowToeNeed(changed==18u,"expected 16 toe and two foot shapes");
 }
"""
src="#include <cstdlib>\n"+old.replace(needle,insert+needle);(B/"c_api.cpp").write_text(src);(A/"host.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),src.splitlines(True),fromfile="a/contact-observer.cpp",tofile="b/traction-observer.cpp")))
lines=subprocess.check_output(["/opt/homebrew/bin/ninja","-C",str(R/"build"),"-t","commands"],text=True).splitlines();line=next(x for x in lines if " -c "+str(R/"src/c_api.cpp") in x);args=shlex.split(line);args[1:1]=["-I"+str(W/"include"),"-I"+str(R/"artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111"),"-I"+str(R/"artifacts/crow-clearance-observer-20261109")];obj=B/"c_api.cpp.o"
for k,v in [("-o",str(obj)),("-MF",str(obj)+".d"),("-MT",str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/"c_api.cpp");subprocess.run(args,cwd=R/"build",check=True)
line=next(x for x in lines if x.startswith(": && /usr/bin/c++") and "-dynamiclib" in x and " -o lib/libmetalrobo.dylib " in x);args=shlex.split(line[len(": && "):-len(" && :")]);args[args.index("-o")+1]=str(B/"libmetalrobo.dylib");args[args.index("CMakeFiles/metalrobo.dir/src/c_api.cpp.o")]=str(obj)
files=["src/core/CompiledMetalMultiArticulatedProgram.cpp","src/core/CompiledMetalMultiArticulatedContactProgram.cpp","src/metal/MetalArticulatedInverseMass.mm","src/metal/MetalArticulatedABA.mm","src/metal/MetalWorld.mm","src/metal/MetalMultiArticulatedContact.mm","src/metal/MetalMultiArticulatedABA.mm","src/metal/MetalMultiArticulatedInverseMass.mm"]
for i,n in enumerate(files):
 key="CMakeFiles/metalrobo.dir/"+n+".o";assert key in args;args[args.index(key)]=str(H/"MetalWorld.mm.o" if i==4 else W/(str(i)+".o"))
subprocess.run(args,cwd=R/"build",check=True);h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/"hashes.json").write_text(json.dumps({"host":h(B/"libmetalrobo.dylib"),"baseline_observer_host":h(S/"libmetalrobo.dylib"),"shader":h(H/"sham.metallib")},indent=2)+"\n")
