from pathlib import Path
import json,subprocess,time,hashlib,os,resource
R=Path("/Users/numi/Work/skin-fat-native-integration-209");S=Path(__file__).resolve().parent
sources=["matter/src/numi_human.mm","matter/src/immutable_shared_contact_storage.hpp","apps/numilab_human_myosim_visual_probe.mm","tests/numi_human_fem_surface_attachment/run.py","matter/include/numi/matter/fem_body_contact_sample_grouping.hpp","tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_math_test.cpp","tests/numi_human_fem_surface_attachment/native_initialize_test.mm","tests/numi_human_fem_surface_attachment/immutable_shared_contact_storage_test.mm"]
def pin(p):return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
pre={p:pin(R/p) for p in sources}; stages=[]
def run(name,args):
 t=time.monotonic()
 with (S/(name+".stdout.log")).open("w") as o,(S/(name+".stderr.log")).open("w") as e:
  process=subprocess.Popen(list(map(str,args)),cwd=R,stdout=o,stderr=e)
  (S/"active-build-control.json").write_text(json.dumps({"stage":name,"pid":process.pid,"argv":list(map(str,args))},indent=2)+"\n")
  rc=process.wait()
 record={"name":name,"argv":list(map(str,args)),"exit_code":rc,"elapsed_s":time.monotonic()-t,"stdout":pin(S/(name+".stdout.log")),"stderr":pin(S/(name+".stderr.log"))};stages.append(record)
 (S/"controls.json").write_text(json.dumps({"source_pins":pre,"stages":stages,"physical_admission":False},indent=2)+"\n")
 print(name,rc,record["elapsed_s"],flush=True)
 if rc:raise RuntimeError(name+" failed")
run("native-build",["/opt/homebrew/bin/cmake","--build",R/"build-native","--target","numi-human-native","--parallel","1"])
cpp=["xcrun","--sdk","macosx","clang++","-std=c++23","-O3","-Wall","-Wextra","-Wpedantic","-Werror","-DNDEBUG","-arch","arm64","-fobjc-arc","-I",R/"include","-I",R/"matter/include","-I",R/"matter/src"]
frameworks=["-framework","Foundation","-framework","Metal"]
run("storage-control-build",cpp+[R/"tests/numi_human_fem_surface_attachment/immutable_shared_contact_storage_test.mm",*frameworks,"-o",S/"storage-test"])
run("storage-control",[S/"storage-test"])
lib=R/"build-native/lib/libmetalrobo.dylib";metal=R/"build-native/matter/shaders/NumiMatter.metallib"
run("grouping-control-build",[x for x in cpp if x != "-fobjc-arc"]+[R/"tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_math_test.cpp","-o",S/"grouping-test"])
run("grouping-control",[S/"grouping-test"])
run("native-attachment-control-build",cpp+[R/"tests/numi_human_fem_surface_attachment/native_initialize_test.mm",lib,"-Wl,-rpath,"+str(lib.parent),*frameworks,"-o",S/"native-test"])
run("native-attachment-control",[S/"native-test",R/"matter/materials/silicone.nmatter",metal])
run("native-attachment-active-contact-control",[S/"native-test",R/"matter/materials/silicone.nmatter",metal,"active-contact"])
post={p:pin(R/p) for p in sources};assert pre==post
(S/"build-source-device-manifest.json").write_text(json.dumps({"status":"PASS","source_pins":pre,"source_unchanged_during_build_and_tests":True,"stages":stages,"binary":pin(R/"build-native/bin/numi-human-native"),"runtime_dependencies":{str(p):pin(p) for p in [lib,metal]},"physical_admission":False,"accepted_full_human_steps":0,"maximum_child_resident_bytes":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},indent=2)+"\n")
