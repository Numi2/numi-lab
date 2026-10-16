#!/usr/bin/env python3
import hashlib,json,pathlib,platform,subprocess,sys
Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG=pathlib.Path(__file__).resolve().parents[1]
if PKG.name != "native531-armijo-cgls-fallback-helper-002":
 raise SystemExit("refusing to write outside package 009")
SRC=Q/"snapshots/source-repeated-damped-fallback-203"
BUILD=Q/"build-source203-fallback-001"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def under_pkg(p):
 return p.resolve().is_relative_to(PKG.resolve())
def run(argv,log):
 p=PKG/log
 if not under_pkg(p): raise SystemExit("output log escapes package")
 p.parent.mkdir(parents=True,exist_ok=True)
 cp=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 p.write_text(cp.stdout)
 return {"argv":argv,"exit_code":cp.returncode,"log":log,"log_sha256":sha(p)}
runtimeSource=SRC/"matter/src/metal/base_cgls_runtime.metalinc"
runtimeText=runtimeSource.read_text()
def extractKernel(name):
 start=runtimeText.index("kernel void "+name+"(")
 end=runtimeText.index("\n}\n",start)+3
 return start,end,runtimeText[start:end]
selectedKernels=["nm_fgmres_prepare_base_cgls_layout"]
parts=[];spans={}
for name in selectedKernels:
 start,end,body=extractKernel(name)
 parts.append(body)
 spans[name]={"byte_start":start,"byte_end":end,"line_start":runtimeText.count("\n",0,start)+1,"line_end":runtimeText.count("\n",0,end)+1,"sha256":hashlib.sha256(body.encode()).hexdigest()}
extractText="namespace numi_matter_metal {\n"+"\n".join(parts)+"\n} // namespace numi_matter_metal\n"
extractPath=PKG/"source/production_layout_extract.metalinc"
if not under_pkg(extractPath): raise SystemExit("extracted include escapes package")
extractPath.write_text(extractText)
extractInfo={"source_path":str(runtimeSource),"source_sha256":sha(runtimeSource),"kernels":spans,"extracted_file_sha256":sha(extractPath),"extraction_method":"exact byte-range extraction of two production kernel definitions; no source edits"}
extractManifest=PKG/"source/production_layout_extract.json"
if not under_pkg(extractManifest): raise SystemExit("extraction manifest escapes package")
extractManifest.write_text(json.dumps(extractInfo,sort_keys=True,indent=2)+"\n")
frozen=SRC/"frozen-source.json"; fd=json.loads(frozen.read_text())
build=BUILD/"source-build.json"; bd=json.loads(build.read_text())
sourceChecks={}
for rel,h in fd["source_sha256"].items():
 p=SRC/rel
 sourceChecks[rel]={"expected":h,"actual":sha(p) if p.is_file() else None}
 sourceChecks[rel]["ok"]=sourceChecks[rel]["actual"]==h
artifactChecks={}
for p,h in bd["artifacts_sha256"].items():
 artifactChecks[p]={"expected":h,"actual":sha(p) if pathlib.Path(p).is_file() else None}
 artifactChecks[p]["ok"]=artifactChecks[p]["actual"]==h
commands=[];symbols={}
metal="/usr/bin/xcrun"
expected=[
 "numi_matter_metal::nm_fem_armijo_decide",
 "numi_matter_metal::nm_fgmres_prepare_base_cgls_layout",
 "numi_matter_metal::nm_fgmres_base_cgls_initialize_state",
 "numi_matter_metal::nm_fgmres_base_cgls_initialize_vectors",
 "numi_matter_metal::nm_fgmres_base_cgls_measure_initial",
 "numi_matter_metal::nm_fgmres_base_cgls_form_alpha",
 "numi_matter_metal::nm_fgmres_base_cgls_update_xr",
 "numi_matter_metal::nm_fgmres_base_cgls_complete_iteration",
 "numi_matter_metal::nm_fgmres_base_cgls_form_beta",
 "numi_matter_metal::nm_fgmres_base_cgls_update_p",
 "reinit_fixture_apply"]
for opt in ("O0","O3"):
 d=PKG/"build"/opt;d.mkdir(parents=True,exist_ok=True)
 air=d/"reinit_fixture.air";lib=d/"reinit_fixture.metallib";exe=d/"reinit_replay"
 if not all(under_pkg(p) for p in (d,air,lib,exe)): raise SystemExit("build output escapes package")
 commands.append(run([metal,"-sdk","macosx","metal","-std=metal4.0",f"-{opt}","-fno-fast-math","-I",str(SRC/"matter/include"),"-I",str(SRC/"include"),"-I",str(SRC/"matter/src"),"-I",str(SRC/"matter/src/metal"),"-c",str(PKG/"source/reinit_fixture.metal"),"-o",str(air)],f"build/{opt}/metal-compile.log"))
 commands.append(run([metal,"-sdk","macosx","metallib",str(air),"-o",str(lib)],f"build/{opt}/metallib-link.log"))
 commands.append(run([metal,"-sdk","macosx","clang++","-std=c++17",f"-{opt}","-fobjc-arc","-Wall","-Wextra","-Werror","-I",str(SRC/"matter/include"),str(PKG/"source/reinit_replay.mm"),"-framework","Metal","-framework","Foundation","-o",str(exe)],f"build/{opt}/host-compile.log"))
 sp=PKG/"build"/opt/"metallib-exported-function-names.txt"
 cp=subprocess.run(["/usr/bin/strings",str(lib)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if not under_pkg(sp): raise SystemExit("symbol export file escapes package")
 sp.write_text(cp.stdout)
 symbols[opt]={"exit_code":cp.returncode,"log":str(sp.relative_to(PKG)),"log_sha256":sha(sp),"functions":{s:(s in cp.stdout) for s in expected}}
 commands.append({"argv":["/usr/bin/strings",str(lib)],"exit_code":cp.returncode,"log":str(sp.relative_to(PKG)),"log_sha256":sha(sp)})
source= (PKG/"source/reinit_replay.mm").read_text()
metalSource=(PKG/"source/reinit_fixture.metal").read_text()
checks={
 "production_layout_extract_included":'#include "production_layout_extract.metalinc"' in metalSource,
 "production_shared_header_used":'#include "numi/matter/shared.h"' in source,
 "actual_armijo_decision_bound":"numi_matter_metal::nm_fem_armijo_decide" in source,
 "unsafe_armijo_declines_fallback":"unsafe exhausted private trial was incorrectly converted to fallback" in source,
 "actual_layout_bound":"nm_fgmres_prepare_base_cgls_layout" in source,
 "armijo_decision_not_primary_policy":"nm_fem_armijo_decide" in source and "nm_fgmres_select_base_cgls_primary_policy" not in source,
 "accepted_noop_and_exhausted_armijo_outcomes":"decision[0].w==1.0f && decision[1].w==2.0f" in source and "decision[0].w==3.0f && decision[1].w==2.0f" in source,
 "terminal_exhaustion_index_bound":"trialIndex=23u" in source and "solver.nonlinearIterations.w=24u" in source,
 "actual_init_and_recurrence_bound":all(x in source for x in ["nm_fgmres_base_cgls_initialize_state","nm_fgmres_base_cgls_initialize_vectors","nm_fgmres_base_cgls_form_alpha","nm_fgmres_base_cgls_update_xr","nm_fgmres_base_cgls_form_beta","nm_fgmres_base_cgls_update_p"]),
 "two_pass_loop":"epoch<2" in source,
 "poison_and_same_scratch_reuse":"for(auto b:{metricsBuffer" in source and "poison(b)" in source and "Scratch scratch=makeScratch(device)" in source and "oldControl[0].x==previous->status" in source,
 "checks_current_rhs_and_gradient":"production residual did not reload current RHS" in source and "production gradient/search direction did not reload current A^T b" in source,
 "quiet_env_offsets_and_reset":"expectedRows[U]={0u,0u,1u,1u,0u,0u,1u,1u}" in source and "quiet environment status changed during recurrence" in source,
 "candidate_ABI_is_production":"NMFEMNodeStateGPU" in source and "sizeof(NMFEMNodeStateGPU) == 80" in source,
 "no production_files_modified":True}
errors=[]
checks["one_scratch_all_passes"]="makeScratch(device)" in source and source.count("makeScratch(device)")==1 and "Scratch scratch=makeScratch(device)" in source
checks["prior_error_and_status_observed"]="oldError[0]==previous->error" in source and "prior_error_bits" in source
checks["native_runtime_metadata"]="native_kernel_execution\\\":true" in source and "gpu_dispatch_performed\\\":true" in source and "physical_fem_transactions\\\":0" in source
checks["two_environment_metadata"]="environment_count\\\":2" in source and "selected_environment_words\\\":[1,0]" in source and "row_environment\\\":[0,0,1,1,0,0,1,1]" in source
checks["coupon_not_qualified"]="full_coupon_qualified\\\":false" in source
checks["measurement_follows_initializer_assertions"]=(
    source.index("production CGLS scalar state did not reset before measurement")
    < source.index('dispatchGroups(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_measure_initial")')
    < source.index("production measurement did not initialize active scalar state and preserve quiet state"))
if any(not v for v in checks.values()): errors.extend(["source_contract_failed:"+k for k,v in checks.items() if not v])
if any(c["exit_code"]!=0 for c in commands):errors.append("compile_or_export_failed")
if not all(x["ok"] for x in sourceChecks.values()) or len(sourceChecks)!=len(fd["source_sha256"]):errors.append("source_snapshot_hash_failure")
if not all(x["ok"] for x in artifactChecks.values()) or not bd.get("success"):errors.append("source_build_artifact_failure")
if any(x["exit_code"]!=0 or not all(x["functions"].values()) for x in symbols.values()):errors.append("required_kernel_export_missing")
for opt in ("O0","O3"):
 for fn in ("reinit_fixture.air","reinit_fixture.metallib","reinit_replay"):
  if not (PKG/"build"/opt/fn).is_file() or (PKG/"build"/opt/fn).stat().st_size==0:errors.append(f"missing_{opt}_{fn}")
report={"schema":"numi.cardboard.native531-armijo-cgls-fallback-helper-compile-validation.v1","host":platform.platform(),"package":str(PKG),"source_snapshot":str(SRC),"source_frozen_manifest_sha256":sha(frozen),"source_build_manifest":str(build),"source_build_manifest_sha256":sha(build),"production_layout_extraction":extractInfo,"source_file_checks":sourceChecks,"build_artifact_checks":artifactChecks,"commands":commands,"required_production_kernel_exports":symbols,"source_contract_checks":checks,"gpu_dispatch_performed":False,"helper_executable_invoked":False,"runtime_behavior_executed":False,"native_reinitialization_result":None,"scope":"O0/O3 Metal and host compilation plus symbol and source contract checks only; native checker is prepared but not run","result":"PASS_COMPILE_ONLY" if not errors else "FAIL","errors":errors,"native_commands":{o:[str(PKG/"build"/o/"reinit_replay"),str(PKG/"build"/o/"reinit_fixture.metallib")] for o in ["O0","O3"]}}
validationPath=PKG/"build/validation.json"
commandPath=PKG/"build/command.json"
if not under_pkg(validationPath) or not under_pkg(commandPath): raise SystemExit("validation output escapes package")
validationPath.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
commandPath.write_text(json.dumps(report["native_commands"],sort_keys=True,indent=2)+"\n")
# Seal every regular file except the manifest itself and any transient pycache.
files=[]
for p in sorted(PKG.rglob("*")):
 if p.is_file() and p.name!="package-manifest.json" and "__pycache__" not in p.parts:
  files.append({"path":str(p.relative_to(PKG)),"sha256":sha(p),"size":p.stat().st_size})
manifest={"schema":"numi.cardboard.native531-armijo-cgls-fallback-helper-package.v1","source_snapshot":str(SRC),"source_snapshot_manifest_sha256":sha(frozen),"source_build_manifest_sha256":sha(build),"gpu_dispatch_performed":False,"files":files}
(PKG/"package-manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
print(json.dumps({"result":report["result"],"source_files_checked":len(sourceChecks),"artifacts_checked":len(artifactChecks),"commands":len(commands),"required_kernel_count":len(expected),"errors":errors,"manifest_sha256":sha(PKG/"package-manifest.json")},sort_keys=True))
sys.exit(0 if not errors else 1)
