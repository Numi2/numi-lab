import os,json,pathlib,subprocess,time,hashlib,sys
r=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
e=r/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020"
d=json.load(open(e/"metal-optimization-073/run/invocation.json"))
mode=sys.argv[1]; out=e/"metal-optimization-078"/mode;out.mkdir(parents=True,exist_ok=False)
a=d["argv"];a[0]=str(r/"build-integrated-validation-20261019/perf078-build27-frozen/numi-human-native");a[4]=str(out/"output");a[a.index("--muscle-step-count")+1]=sys.argv[2] if len(sys.argv)>2 else "1"
env=os.environ.copy();env["DYLD_LIBRARY_PATH"]=str(r/"build-integrated-validation-20261019/perf078-build27-frozen");env.update(d["simulation_environment"]);env["NUMI_MATTER_PIPELINE_TRACE"]="1"
for k in list(env):
 if k.startswith(("NM_FGMRES_","NM_CONTACT_NODE_INCIDENCE_","NUMI_MATTER_GPU_TIMING","NUMI_MATTER_NEWTON_PRECONDITIONER","NM_FEM_DISTRIBUTED_GLOBAL","NM_FEM_CANONICAL_NEOHOOKEAN","NM_FEM_SHARED_LOCAL_FACTORIZATION")):env.pop(k)
if mode.startswith("control"):
 p=r/"build-integrated-validation-20261019/perf075-control";a[0]=str(p/"numi-human-native");env["DYLD_LIBRARY_PATH"]=str(p)
else:
 env.update(NM_FGMRES_DISTRIBUTED_CGS2="1",NM_CONTACT_NODE_INCIDENCE_RADIX="1",NUMI_MATTER_NEWTON_PRECONDITIONER_REUSE="1")
 if "icb" in mode:env["NM_FGMRES_CGS2_ICB"]="1"
 if "global" in mode:env["NM_FEM_DISTRIBUTED_GLOBAL_MEASUREMENTS"]="1"
 if "profile" in mode:env.update(NUMI_MATTER_WORK_COUNTERS="1",NUMI_MATTER_GPU_TIMING="1",NUMI_MATTER_GPU_TIMING_STAGES="1",NUMI_MATTER_GPU_TIMING_ROOT="0",NUMI_MATTER_GPU_TIMING_KERNELS="all",NUMI_MATTER_GPU_TIMING_SAMPLE_EVERY="127")
if "stages" in mode:
 env.update(NUMI_MATTER_WORK_COUNTERS="1",NUMI_MATTER_GPU_TIMING="1",NUMI_MATTER_GPU_TIMING_STAGES="1",NUMI_MATTER_GPU_TIMING_ROOT="0",NUMI_MATTER_GPU_TIMING_KERNELS="none")
if "cachefull" in mode:env.update(NUMI_MATTER_GPU_TIMING_KERNELS="nm_fem_build_algorithmic_tangent_cache_columns",NUMI_MATTER_GPU_TIMING_SAMPLE_EVERY="1")
if "cachecoop" in mode:env.update(NUMI_MATTER_GPU_TIMING_KERNELS="nm_fem_build_algorithmic_tangent_cache_bundle_cooperative",NUMI_MATTER_GPU_TIMING_SAMPLE_EVERY="1")
if "operators" in mode:env.update(NUMI_MATTER_GPU_TIMING_KERNELS="nm_fem_apply_operator_elements,nm_fgmres_gather_nodes,nm_vascular_operator,nm_vascular_cavity_operator",NUMI_MATTER_GPU_TIMING_SAMPLE_EVERY="1")
if "work" in mode:env["NUMI_MATTER_WORK_COUNTERS"]="1"
if "ew" in mode:env["NM_FGMRES_INEXACT_NEWTON_FORCING"]="1"
if "shared" in mode:env["NM_FEM_SHARED_LOCAL_FACTORIZATION"]="1"
if "scalar" in mode:env["NUMI_HUMAN_MATERIAL_EXPRESSION_CACHE"]="1"
else:env["NUMI_HUMAN_MATERIAL_EXPRESSION_CACHE"]="0"
if "census" in mode:env["NM_FEM_TANGENT_CENSUS"]="1"
if "canonical" in mode:env["NM_FEM_CANONICAL_NEOHOOKEAN_TANGENT"]="1"
if "root3" in mode:env["NUMI_MATTER_GPU_TIMING_ROOT"]="3"
if "phase" in mode:env["NUMI_MATTER_GPU_TIMING_PHASE"]=mode.split("phase")[-1].split("-")[0]
env["NUMI_HUMAN_MATERIAL_EXPRESSION_CACHE"]="1"
env["NM_FEM_STATELESS_TANGENT_BUNDLE"]="2" if "coop" in mode else ("1" if "bundle" in mode or "native" in mode or "generate" in mode else "0")
env["NUMI_MATTER_ASYNC_PIPELINES"]="1"
env["NUMI_MATTER_METALLIB_OVERRIDE"]=str(r/"build-integrated-validation-20261019/perf078-build27-frozen/NumiMatter.metallib")
if "hier" in mode:env["NM_CONTACT_NODE_INCIDENCE_HIERARCHICAL_PREFIX"]="1"
if "witness" in mode:env["NM_FEM_NATIVE_TANGENT_WITNESS_TET"]="3315"
if "materialpso" in mode:env["NM_FEM_MATERIAL_PIPELINES"]="1"
if "femtangent" in mode:env["NUMI_MATTER_HUMAN_PARTITIONED_FEM_TANGENT"]="1"
if "generate" in mode:env["NM_FEM_WRITE_NATIVE_TANGENT_SOURCE"]=str(r/"matter/src/metal/generated_stateless_tangent.metalinc")
if "native" in mode:env["NM_FEM_NATIVE_TANGENT_BUNDLE"]="1"
if "coarse" in mode:env["NUMI_MATTER_HUMAN_PARTITIONED_PRECONDITIONER"]="1"
if "diag" in mode:env["NUMI_HUMAN_PARTITIONED_REDUCTION_DIAGNOSTIC"]="1"
if "archive" in mode:env["NUMI_MATTER_PIPELINE_ARCHIVE"]=str(r/"build-integrated-validation-20261019/perf078-pipelines.metalar")
if "tail" in mode:env["NM_FEM_NEWTON_TAIL_DISPATCH"]="1"
if "tailicb" in mode:env["NM_FEM_NEWTON_TAIL_ICB"]="1"
if "operators" in mode:env["NUMI_MATTER_GPU_TIMING_CAPACITY"]="65536"
if "stateless" in mode:env["NM_FEM_STATELESS_MATERIAL_SPECIALIZATION"]="1"
if "verify" in mode:env["NM_FEM_NATIVE_TANGENT_VERIFY"]="1"
if "allnative" in mode:env["NM_FEM_ALL_NATIVE_FLOAT_SPECIALIZATION"]="1"
if "relative" in mode:env["NM_FGMRES_RELATIVE_INNER_FORCING"]="1"
if "springs" in mode:env["NUMI_HUMAN_IMPLICIT_BODY_SPRINGS"]="20"
if "generate" in mode:env["NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY"]="1"
if "block" in mode:env["NUMI_HUMAN_RESTING_COUPLED_TANGENT_BLOCK"]="1"
if "-graph-" in mode:env["NM_FEM_HUMAN6_GRAPH_SCHWARZ"]="1"
if "domain" in mode:env["NM_FEM_DOMAIN_NONLINEAR_LATCH"]="1"
if "tight2" in mode:env["NM_FEM_DOMAIN_MECHANICAL_TIGHTENING"]="2"
if "operatorgraph" in mode:env["NM_FGMRES_OPERATOR_ICB"]="1"
if "hybrid" in mode:env["NM_FEM_HUMAN_HYBRID_COORDINATES"]="1"
if "ewtight1" in mode:env["NM_FGMRES_INEXACT_NEWTON_TIGHTENING"]="1"
if "ewtight2" in mode:env["NM_FGMRES_INEXACT_NEWTON_TIGHTENING"]="2"
if "tight3" in mode:env["NM_FEM_DOMAIN_MECHANICAL_TIGHTENING"]="3"
if "laggraph" in mode:env["NM_FEM_HUMAN6_GRAPH_FACTOR_LAG"]="1"
if "recycle" in mode:env["NM_FEM_REGIONAL_SELECTOR6_ACCEPTED_Z0_RECYCLE"]="1"
if "row16" in mode:env["NUMI_MATTER_VASCULAR_ROW16_DIRECTION_TRACE"]="1"
if "cavity3" in mode:env["NUMI_MATTER_VASCULAR_BLOCK_CAVITY_SCHUR"]="1"
if "principal" in mode:env["NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER"]="1"
if "predictor" in mode:env["NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR"]="1"
if "supported" in mode:
 import re
 match=re.search(r"-dt(\d+)us",mode)
 dt=float(match.group(1))*1e-6 if match else .000125
 a[a.index("--muscle-step-seconds")+1]=format(dt,".9f")
 iz=a.index("--root-pose")+3
 a[iz]=format(float(a[iz])+.0001012,".12g")
 if "--resting-gravity-ramp-steps" in a:
  ir=a.index("--resting-gravity-ramp-steps");del a[ir:ir+2]
 a.extend(["--resting-gravity-ramp-steps",str(round(.004/dt)+1)])
 env["NUMI_HUMAN_STATIC_WEIGHTED_SKIN_SUPPORT"]="1"
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-field-failure.json")
record={k:v for k,v in d.items() if k not in ("working_source_sha256","working_tree_diff_sha256")};record.update(cwd=str(out),argv=a,simulation_environment={k:v for k,v in env.items() if k.startswith(("NM_","NUMI_","DYLD_"))},binary_sha256=hashlib.sha256(open(a[0],"rb").read()).hexdigest())
record["source_head"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=r,text=True).strip()
record["working_source_sha256"]={str(f.relative_to(r)):hashlib.file_digest(open(f,"rb"),"sha256").hexdigest() for f in [r/"matter/tools/human_resting_runtime.hpp",r/"matter/include/numi/matter/paper_compensated_bytecode.h",r/"matter/src/compiler_impl.cpp",r/"matter/src/constitutive_impl.cpp",r/"matter/src/validation_impl.cpp",r/"matter/src/metal/common.metalinc",r/"matter/tools/fem_material_frame_check.mm",r/"matter/src/runtime.mm",r/"matter/src/native_tangent_source.hpp",r/"matter/src/metal/generated_stateless_tangent.metalinc",r/"matter/src/metal_encoder_timing.hpp",r/"matter/src/metal/fem.metalinc",r/"matter/src/metal/fgmres.metalinc",r/"matter/src/metal/contact.metalinc",r/"matter/include/numi/matter/shared.h"]}
record["artifact_sha256"]={str(f):hashlib.file_digest(open(f,"rb"),"sha256").hexdigest() for f in ([p/"libmetalrobo.dylib",p/"NumiMatter.metallib"] if mode.startswith("control") else [r/"build-integrated-validation-20261019/perf078-build27-frozen/libmetalrobo.dylib",r/"build-integrated-validation-20261019/perf078-build27-frozen/NumiMatter.metallib"])}
record["working_source_sha256"]=json.load(open(e/"metal-optimization-078/native-build27-source-manifest.json"))
record["host_source_manifest"]="native-build27-source-manifest.json"
record["shader_source_manifest"]="native-build27-source-manifest.json"
record["cohort_note"]="Normal full CMake build27; full source changes archived in perf078-build27-frozen."
json.dump(record,open(out/"invocation.json","w"),indent=2)
t=time.monotonic()
with open(out/"stdout.log","w") as so,open(out/"stderr.log","w") as se:
 try:code=subprocess.run(a,cwd=str(out),env=env,stdout=so,stderr=se,timeout=600).returncode
 except subprocess.TimeoutExpired:code="censored_600s"
json.dump(dict(exit_code=code,process_wall_seconds=time.monotonic()-t),open(out/"process.json","w"),indent=2)
print(mode,code,time.monotonic()-t)

if code!=0:sys.exit(1)
