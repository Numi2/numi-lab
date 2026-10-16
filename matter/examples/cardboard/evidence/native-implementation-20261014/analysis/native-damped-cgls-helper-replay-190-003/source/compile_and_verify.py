#!/usr/bin/env python3
import hashlib,json,pathlib,platform,subprocess,sys
Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG=Q/"analysis/native-damped-cgls-helper-replay-190-003"
SRC=Q/"snapshots/source-damped-full-zero-cgls-190-001"
BUILD=Q/"build-source190-002"
OUT=PKG/"build"
PYTHON="/opt/homebrew/opt/python@3.13/bin/python3.13"
EXPECT=[
 "numi_matter_metal::nm_fgmres_base_cgls_initialize_state",
 "numi_matter_metal::nm_fgmres_base_cgls_initialize_vectors",
 "numi_matter_metal::nm_fgmres_base_cgls_measure_initial",
 "numi_matter_metal::nm_fgmres_base_cgls_form_alpha",
 "numi_matter_metal::nm_fgmres_base_cgls_update_xr",
 "numi_matter_metal::nm_fgmres_base_cgls_complete_iteration",
 "numi_matter_metal::nm_fgmres_base_cgls_form_beta",
 "numi_matter_metal::nm_fgmres_base_cgls_update_p",
 "damped_fixture_apply"]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def run(name,argv):
 cp=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log=OUT/name
 log.parent.mkdir(parents=True,exist_ok=True)
 log.write_text(cp.stdout)
 return {"argv":argv,"exit_code":cp.returncode,"log":str(log.relative_to(PKG)),"log_sha256":sha(log),"output":cp.stdout}
commands=[]
metal="/usr/bin/xcrun"
export_checks={}
for opt in ("O0","O3"):
 d=OUT/opt
 d.mkdir(parents=True,exist_ok=True)
 air=d/"damped_cgls_fixture.air"
 lib=d/"damped_cgls_fixture.metallib"
 exe=d/"damped_cgls_helper_replay"
 commands.append(run(f"{opt}/metal-compile.log",[metal,"-sdk","macosx","metal","-std=metal4.0",f"-{opt}","-fno-fast-math","-I",str(SRC/"matter/include"),"-I",str(SRC/"include"),"-I",str(SRC/"matter/src"),"-I",str(SRC/"matter/src/metal"),"-c",str(PKG/"source/damped_cgls_fixture.metal"),"-o",str(air)]))
 commands.append(run(f"{opt}/metallib-link.log",[metal,"-sdk","macosx","metallib",str(air),"-o",str(lib)]))
 commands.append(run(f"{opt}/host-compile.log",[metal,"-sdk","macosx","clang++","-std=c++17",f"-{opt}","-fobjc-arc","-Wall","-Wextra","-Werror",str(PKG/"source/damped_cgls_helper_replay.mm"),"-framework","Metal","-framework","Foundation","-o",str(exe)]))
 symbols=subprocess.run(["/usr/bin/strings",str(lib)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log=OUT/opt/"metallib-exported-function-names.txt"
 log.write_text(symbols.stdout)
 found={s:(s in symbols.stdout) for s in EXPECT}
 export_checks[opt]={"exit_code":symbols.returncode,"log":str(log.relative_to(PKG)),"log_sha256":sha(log),"expected_symbols":found,"all_expected_present":symbols.returncode==0 and all(found.values())}
 commands.append({"argv":["/usr/bin/strings",str(lib)],"exit_code":symbols.returncode,"log":str(log.relative_to(PKG)),"log_sha256":sha(log)})
oracle=subprocess.run([PYTHON,str(PKG/"source/cpu_oracle.py")],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(OUT/"cpu-oracle.json").write_text(oracle.stdout)
commands.append({"argv":[PYTHON,str(PKG/"source/cpu_oracle.py")],"exit_code":oracle.returncode,"log":"build/cpu-oracle.json","log_sha256":sha(OUT/"cpu-oracle.json")})
frozen=SRC/"frozen-source.json"
fd=json.loads(frozen.read_text())
source_checks={}
for rel,expected in fd["source_sha256"].items():
 p=SRC/rel
 if p.is_file():
  actual=sha(p)
  source_checks[rel]={"expected":expected,"actual":actual,"ok":actual==expected}
build_doc=json.loads((BUILD/"source-build.json").read_text())
artifact_checks={}
for name,expected in build_doc["artifacts_sha256"].items():
 p=pathlib.Path(name)
 actual=sha(p) if p.is_file() else None
 artifact_checks[name]={"expected":expected,"actual":actual,"ok":actual==expected}
errors=[]
if any(c["exit_code"]!=0 for c in commands): errors.append("compile_or_static_export_or_cpu_oracle_failed")
if not fd.get("schema")=="numi.cardboard.frozen-source.v1": errors.append("unexpected_frozen_source_schema")
if not build_doc.get("success"): errors.append("source190_build_not_successful")
if not all(v["ok"] for v in source_checks.values()) or len(source_checks)!=len(fd["source_sha256"]): errors.append("source_hash_mismatch_or_missing_source")
if not all(v["ok"] for v in artifact_checks.values()): errors.append("source190_build_artifact_mismatch")
if not all(v["all_expected_present"] for v in export_checks.values()): errors.append("expected_metal_function_export_missing")
if oracle.returncode or json.loads(oracle.stdout).get("result")!="PASS": errors.append("cpu_oracle_failed")
for opt in ("O0","O3"):
 for f in ("damped_cgls_fixture.air","damped_cgls_fixture.metallib","damped_cgls_helper_replay"):
  if not (OUT/opt/f).is_file() or (OUT/opt/f).stat().st_size==0: errors.append(f"missing_{opt}_{f}")
report={"schema":"numi.cardboard.damped-cgls-helper-compile-validation.v2","host":platform.platform(),"source_snapshot":str(SRC),"frozen_source_sha256":sha(frozen),"source_build_manifest":str(BUILD/"source-build.json"),"source_build_manifest_sha256":sha(BUILD/"source-build.json"),"source_build_success":build_doc.get("success"),"source_file_hash_checks":source_checks,"source_build_artifact_checks":artifact_checks,"metal_function_export_checks":export_checks,"commands":commands,"gpu_dispatch_performed":False,"helper_executable_invoked":False,"native_kernel_replay_result":None,"result":"PASS" if not errors else "FAIL","errors":errors}
(OUT/"validation.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"result":report["result"],"commands":len(commands),"source_files_checked":len(source_checks),"source_artifacts_checked":len(artifact_checks),"qualified_exports_checked":len(EXPECT),"errors":errors},sort_keys=True))
sys.exit(0 if not errors else 1)
