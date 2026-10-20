import os,json,pathlib,subprocess,time,hashlib,sys
r=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
e=r/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020"
d=json.load(open(e/"metal-optimization-073/run/invocation.json"))
mode=sys.argv[1]; out=e/"metal-optimization-075"/mode;out.mkdir(parents=True,exist_ok=True)
a=d["argv"];a[4]=str(out/"output");a[a.index("--muscle-step-count")+1]=sys.argv[2] if len(sys.argv)>2 else "1"
env=os.environ.copy();env.update(d["simulation_environment"]);env["NUMI_MATTER_PIPELINE_TRACE"]="1"
for k in list(env):
 if k.startswith(("NM_FGMRES_","NM_CONTACT_NODE_INCIDENCE_","NUMI_MATTER_GPU_TIMING","NUMI_MATTER_NEWTON_PRECONDITIONER","NM_FEM_DISTRIBUTED_GLOBAL")):env.pop(k)
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
if "root3" in mode:env["NUMI_MATTER_GPU_TIMING_ROOT"]="3"
if "phase" in mode:env["NUMI_MATTER_GPU_TIMING_PHASE"]=mode.split("phase")[-1].split("-")[0]
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-field-failure.json")
record={k:v for k,v in d.items() if k not in ("working_source_sha256","working_tree_diff_sha256")};record.update(cwd=str(out),argv=a,simulation_environment={k:v for k,v in env.items() if k.startswith(("NM_","NUMI_","DYLD_"))},binary_sha256=hashlib.sha256(open(a[0],"rb").read()).hexdigest())
record["working_source_sha256"]={str(f.relative_to(r)):hashlib.file_digest(open(f,"rb"),"sha256").hexdigest() for f in [r/"matter/src/runtime.mm",r/"matter/src/metal_encoder_timing.hpp",r/"matter/src/metal/fem.metalinc",r/"matter/src/metal/fgmres.metalinc",r/"matter/src/metal/contact.metalinc",r/"matter/include/numi/matter/shared.h"]}
record["artifact_sha256"]={str(f):hashlib.file_digest(open(f,"rb"),"sha256").hexdigest() for f in ([p/"libmetalrobo.dylib",p/"NumiMatter.metallib"] if mode.startswith("control") else [r/"build-integrated-validation-20261019/profile-build-039/lib/libmetalrobo.dylib",r/"build-integrated-validation-20261019/profile-build-039/matter/shaders/NumiMatter.metallib"])}
json.dump(record,open(out/"invocation.json","w"),indent=2)
t=time.monotonic()
with open(out/"stdout.log","w") as so,open(out/"stderr.log","w") as se:
 try:code=subprocess.run(a,cwd=str(out),env=env,stdout=so,stderr=se,timeout=180).returncode
 except subprocess.TimeoutExpired:code="censored_180s"
json.dump(dict(exit_code=code,process_wall_seconds=time.monotonic()-t),open(out/"process.json","w"),indent=2)
print(mode,code,time.monotonic()-t)
