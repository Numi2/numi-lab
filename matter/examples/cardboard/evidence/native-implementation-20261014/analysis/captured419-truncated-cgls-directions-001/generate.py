#!/usr/bin/env python3
"""Generate bounded CGLS direction proposals for exact source125 capture 419.

CPU-only; A and A^T act on the bound unsymmetrized captured CSR. The generated
vectors are diagnostic proposals, not certified Newton corrections.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, struct, sys
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy import sparse

B=Path("/Users/home/cardboard-implementation-20261014")
HERE=Path(__file__).resolve().parent
PKG=B/"analysis/captured419-regularized-directions-001"
CAP=B/"solver/two-stage-first-failure-rich-fd-419/operator-capture"
CAP_PKG=B/"analysis/captured-base-state-cauchy-419-001"
SOURCE=B/"snapshots/source-status5-vtccd-diagnostics-125"
ASSEMBLY=SOURCE/"matter/tools/cardboard_cached_operator_audit.py"
BASE_PAYLOAD=PKG/"base-replay-input.bin"
HELPER=B/"analysis/native-residual-direction-replay-419-001"
PACK=B/"solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack"
SOURCE_BUILD=Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json")
HEADER=struct.Struct("<8s6IQfI")
TARGETS=(8,16,32,64,128)
PRECISIONS=("fp64","emulated_fp32")

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""): h.update(c)
 return h.hexdigest()

def norm(x):
 a=np.asarray(x,dtype=np.float64).reshape(-1)
 return math.sqrt(max(0.0,math.fsum(float(v)*float(v) for v in a)))

def dot64(a,b):
 aa=np.asarray(a,dtype=np.float64).reshape(-1); bb=np.asarray(b,dtype=np.float64).reshape(-1)
 return math.fsum(float(x)*float(y) for x,y in zip(aa,bb))

def dot32_ordered(a,b):
 aa=np.asarray(a,dtype=np.float32).reshape(-1); bb=np.asarray(b,dtype=np.float32).reshape(-1)
 acc=np.float32(0.0)
 for x,y in zip(aa,bb):
  acc=np.float32(acc+np.float32(x*y))
 return acc

def import_assembly():
 spec=importlib.util.spec_from_file_location("source125_cgls_assembly",ASSEMBLY)
 if spec is None or spec.loader is None: raise RuntimeError("cannot import source125 assembly")
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def req(ok,msg):
 if not ok: raise AssertionError(msg)

def verify_proposal_inputs():
 m=json.loads((PKG/"manifest.json").read_text())
 req(sha(PKG/"manifest.json")=="339ae206918d3ef99d0a42859de94948299c39ea2114afd30981689c62acc734","captured419 proposal manifest changed")
 for row in m["artifacts"]:
  q=PKG/row["path"];req(q.is_file() and q.stat().st_size==row["bytes"] and sha(q)==row["sha256"],"proposal artifact mismatch "+row["path"])
 require=(m["source_input_bindings"])
 for row in require.values():
  q=Path(row["path"]);req(q.is_file() and q.stat().st_size==row["bytes"] and sha(q)==row["sha256"],"source input changed "+str(q))
 req(sha(CAP)==m["capture_sha256"],"source capture changed")
 req(sha(PACK)==m["compiled_package_sha256"],"compiled package changed")
 req(sha(ASSEMBLY)==require["source125_assembly_module"]["sha256"],"assembly source changed")
 program=json.loads((HELPER/"compiled-program.json").read_text())
 for rel,h in program["files"].items():req(sha(HELPER/rel)==h,"native helper artifact changed "+rel)
 source_build=json.loads(SOURCE_BUILD.read_text())
 req(sha(SOURCE_BUILD)==m["source125_build_sha256"],"source125 build manifest changed")
 for name,h in source_build["artifacts_sha256"].items():req(sha(Path(name))==h,"source125 artifact changed "+name)
 return m,program,source_build

def cgls(A,b,targets,precision):
 maxiter=max(targets)
 if precision=="fp64":
  mat=A.tocsr(); rhs=np.asarray(b,dtype=np.float64)
  dproduct=dot64
  cast=lambda v:np.asarray(v,dtype=np.float64)
 else:
  mat=A.astype(np.float32).tocsr(); rhs=np.asarray(b,dtype=np.float32)
  dproduct=dot32_ordered
  cast=lambda v:np.asarray(v,dtype=np.float32)
 x=cast(np.zeros(mat.shape[1]))
 r=rhs.copy()
 s=cast(mat.T.tocsr()@r)
 p=s.copy()
 gamma=dproduct(s,s)
 traces={}
 breakdown=None
 for iteration in range(1,maxiter+1):
  q=cast(mat@p)
  denom=dproduct(q,q)
  if not np.isfinite(float(gamma)) or not np.isfinite(float(denom)) or float(denom)<=0 or float(gamma)<0:
   breakdown={"iteration":iteration,"gamma":float(gamma),"denominator":float(denom),"reason":"nonfinite or nonpositive CGLS recurrence scalar"}
   break
  alpha=cast(np.asarray(float(gamma)/float(denom))).reshape(()).item()
  if not math.isfinite(float(alpha)):
   breakdown={"iteration":iteration,"gamma":float(gamma),"denominator":float(denom),"reason":"nonfinite alpha"}
   break
  x=cast(x+cast(alpha)*p)
  r=cast(r-cast(alpha)*q)
  snew=cast(mat.T.tocsr()@r)
  gamma_new=dproduct(snew,snew)
  if not np.isfinite(float(gamma_new)) or float(gamma_new)<0:
   breakdown={"iteration":iteration,"gamma_new":float(gamma_new),"reason":"nonfinite/negative new normal residual"}
   break
  if iteration in targets: traces[iteration]=x.copy()
  if float(gamma)==0.0:
   breakdown={"iteration":iteration,"reason":"zero previous normal residual"}
   break
  if float(gamma_new)==0.0:
   if iteration<maxiter: breakdown={"iteration":iteration,"reason":"normal residual reached exact zero"}
   break
  beta=cast(np.asarray(float(gamma_new)/float(gamma))).reshape(()).item()
  p=cast(snew+cast(beta)*p)
  s=snew;gamma=gamma_new
 return traces,breakdown

def main():
 proposal,program,source_build=verify_proposal_inputs()
 coa=import_assembly()
 header,node_rows,tet_rows,integrity=coa.capture_records(CAP)
 _,_,assembled,_,metadata=coa.validate_and_assemble(header,node_rows,tet_rows,integrity,0)
 A=sparse.load_npz(CAP_PKG/"A_free_csr.npz").tocsr()
 diff=(A-assembled).tocsr();diff.eliminate_zeros()
 req(diff.nnz==0 and A.shape==(2538,2538),"CSR differs from independent source125 reconstruction")
 b=np.fromfile(CAP_PKG/"base_rhs.float64le.bin",dtype="<f8")
 free=np.asarray(metadata["free_nodes"],dtype=np.int64)
 req(len(free)==846 and b.size==A.shape[0],"free-node/RHS dimensions changed")
 base=BASE_PAYLOAD.read_bytes()
 magic,version,E,N,T,S,Q,fingerprint,dt,reserved=HEADER.unpack_from(base)
 req((magic,version,E,N,T,S,Q)==(b"NMFDSWP1",1,2,882,1920,15,59),"base replay header changed")
 off=HEADER.size+E*N*80
 dbytes=E*N*16
 req(len(base)==2195272 and off==141168 and dbytes==28224,"base replay payload layout mismatch")
 fixed=np.setdiff1d(np.arange(N),free)
 bnorm=norm(b)
 newfiles={}
 states={}
 for precision in PRECISIONS:
  traces,breakdown=cgls(A,b,TARGETS,precision)
  states[precision]={}
  for target in TARGETS:
   if target not in traces:
    states[precision][str(target)]={"status":"not_reached","breakdown":breakdown}
    continue
   xiter=traces[target]
   xexport=np.asarray(xiter,dtype=np.float32)
   req(np.all(np.isfinite(xexport)),f"{precision} CGLS iteration {target} has nonfinite exported direction")
   exact_res=b-np.asarray(A@xexport.astype(np.float64)).reshape(-1)
   g=np.asarray(A.T@b).reshape(-1)
   full=np.zeros((E,N,4),dtype="<f4")
   full[0,free,:3]=xexport.reshape(len(free),3)
   req(np.all(full[1]==0) and np.all(full[:,:,3]==0) and np.all(full[0,fixed,:3]==0),"invalid full-node CGLS payload")
   direction=full.tobytes(order="C")
   name=f"{precision}_iter{target:03d}"
   dpath=HERE/"directions"/f"{name}.float32x4.bin"
   dpath.parent.mkdir(parents=True,exist_ok=True);dpath.write_bytes(direction)
   payload=bytearray(base);payload[off:off+dbytes]=direction
   ppath=HERE/"replay-inputs"/f"{name}.input.bin"
   ppath.parent.mkdir(parents=True,exist_ok=True);ppath.write_bytes(payload)
   req(payload[:off]==base[:off] and payload[off+dbytes:]==base[off+dbytes:],"non-direction bytes changed")
   gdotp=dot64(g,xexport)
   rnorm=norm(exact_res)
   # A-transpose residual uses exact FP64 matrix on the exported FP32 direction.
   normal=norm(np.asarray(A.T@exact_res).reshape(-1))
   states[precision][str(target)]={
    "status":"exported","iteration":target,"direction_file":str(dpath.relative_to(HERE)),
    "direction_sha256":sha(dpath),"payload_file":str(ppath.relative_to(HERE)),
    "payload_sha256":sha(ppath),"payload_bytes":ppath.stat().st_size,
    "exported_direction_l2_m_per_s":norm(xexport),"exported_direction_linf_m_per_s":float(np.max(np.abs(xexport))),
    "predicted_free_csr_residual_l2_Ns":rnorm,"predicted_free_csr_residual_linf_Ns":float(np.max(np.abs(exact_res))),
    "predicted_residual_ratio_to_b":rnorm/bnorm,"merit_fraction_drop":1-(rnorm/bnorm)**2,
    "g_dot_p":gdotp,"least_squares_directional_derivative":-gdotp,
    "exact_Atranspose_residual_l2":normal,"descent":bool(gdotp>0),
    "direction_payload_shape":[2,882,4],"quiet_environment_zero":bool(not np.any(full[1])),
    "fixed_nodes_zero":bool(not np.any(full[0,fixed,:3])),"w_lane_zero":bool(not np.any(full[:,:,3]))}
   newfiles[str(dpath.relative_to(HERE))]={"bytes":dpath.stat().st_size,"sha256":sha(dpath)}
   newfiles[str(ppath.relative_to(HERE))]={"bytes":ppath.stat().st_size,"sha256":sha(ppath)}
   # Recurrence residual, if finite, is compared separately to recomputed exact residual.
   if precision=="fp64":
    states[precision][str(target)]["recurrence_residual_note"]="FP64 recurrence state; reported prediction recomputed from exact CSR after FP32 export"
   else:
    states[precision][str(target)]["recurrence_arithmetic"]="float32 CSR matvecs, vectors, updates, and ordered float32 dot accumulation on CPU; not a Metal arithmetic model"
 # Compare exported FP32 directions from both recurrence precisions.
 for target in TARGETS:
  f=states["fp64"].get(str(target),{});s=states["emulated_fp32"].get(str(target),{})
  if f.get("status")=="exported" and s.get("status")=="exported":
   xf=np.fromfile(HERE/f["direction_file"],dtype="<f4").reshape(E,N,4)[0,free,:3].reshape(-1).astype(np.float64)
   xs=np.fromfile(HERE/s["direction_file"],dtype="<f4").reshape(E,N,4)[0,free,:3].reshape(-1).astype(np.float64)
   f["vs_emulated_fp32_exported_direction_l2_m_per_s"]=norm(xf-xs)
   f["vs_emulated_fp32_exported_direction_relative_l2"]=norm(xf-xs)/max(norm(xf),1e-300)
   f["vs_emulated_fp32_residual_l2_difference_Ns"]=abs(f["predicted_free_csr_residual_l2_Ns"]-s["predicted_free_csr_residual_l2_Ns"])
   s["vs_fp64_exported_direction_l2_m_per_s"]=norm(xf-xs)
   s["vs_fp64_exported_direction_relative_l2"]=norm(xf-xs)/max(norm(xf),1e-300)
   s["vs_fp64_residual_l2_difference_Ns"]=abs(f["predicted_free_csr_residual_l2_Ns"]-s["predicted_free_csr_residual_l2_Ns"])
 # Attach baseline and native replay binding for eventual, not yet run, native residual tests.
 baseline={}
 for n in ["raw_cauchy","tikhonov_sigma_0.01","tikhonov_sigma_0.001","tikhonov_sigma_0.0001"]:
  baseline[n]=json.loads((PKG/"report.json").read_text())["results"][n]
 report={
  "schema":"numi.cardboard.captured419-truncated-cgls-directions.v1",
  "classification":"CPU-only exact captured-CSR CGLS proposal generation; not native, physical, accepted, or certificate evidence",
  "algorithm":{"name":"unpreconditioned CGLS","initial_direction":"p=0","damping_lambda":0.0,
    "operator":"exact source125 unsymmetric CSR A and A-transpose; no normal matrix formed",
    "iterations":list(TARGETS),"normal_equations":"A-transpose A p = A-transpose b, applied matrix-free",
    "FP32_emulation":"SciPy CSR float32 A/A-transpose matvec; FP32 vectors/updates; deterministic sequential FP32 dot accumulation; not a GPU/Metal implementation"},
  "source_bindings":{"captured419_proposal_manifest_sha256":sha(PKG/"manifest.json"),
    "base_replay_input_sha256":sha(BASE_PAYLOAD),"capture_sha256":sha(CAP),
    "captured_A_free_csr_sha256":sha(CAP_PKG/"A_free_csr.npz"),
    "captured_b_sha256":sha(CAP_PKG/"base_rhs.float64le.bin"),
    "source125_assembly_sha256":sha(ASSEMBLY),"source125_build_sha256":sha(SOURCE_BUILD),
    "source125_helper_binary_sha256":sha(HELPER/"build/native_residual_direction"),
    "source125_compiled_program_sha256":sha(HELPER/"compiled-program.json"),
    "compiled_package_sha256":sha(PACK),"csr_shape":list(A.shape),"csr_nnz":int(A.nnz),
    "environments":E,"nodes_per_environment":N,"free_nodes":int(len(free)),
    "dt_float32":float(dt),"world_fingerprint_hex":f"{fingerprint:016x}",
    "payload_direction_offset":off,"payload_direction_bytes":dbytes},
  "runs":{"blas_thread_limit":1,"gpu_or_native_run":False,"physical_transactions":0,
    "candidate_payloads_preserve_all_non_direction_bytes":True,"csr_reconstruction_exact":True,
    "outputs":states},
  "regularized_baselines_linear_predictions":baseline,
  "limitations":["No Metal/CGLS execution is included.","FP32 emulation is not bitwise or algorithmically equivalent to Metal parallel reductions.","CGLS uses normal equations and may suffer squared conditioning; no certificate or convergence claim is made.","Exported vectors are alternative correction proposals only; no physical transaction, nonlinear acceptance, CCD, or terminal certificate was run.","Every candidate must be independently replayed through the native production residual and normal acceptance/rollback path before use."],
  "native_replay_commands":{
    "helper":str(HELPER/"build/native_residual_direction"),
    "compiled_package":str(PACK),
    "command_template":"<helper> <compiled_package> <this package replay-inputs/<name>.input.bin> <fresh output directory>",
    "preflight_template":"<helper> <compiled_package> <payload> <fresh output directory> --preflight-only"}
 }
 out=HERE/"report.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 allfiles={}
 for q in sorted(HERE.rglob("*")):
  if q.is_file() and q.name not in ("manifest.json",):
   allfiles[str(q.relative_to(HERE))]={"bytes":q.stat().st_size,"sha256":sha(q)}
 manifest={
  "schema":"numi.cardboard.captured419-truncated-cgls-directions-package.v1",
  "classification":report["classification"],
  "input_manifest_sha256":sha(PKG/"manifest.json"),
  "report_sha256":sha(out),
  "files":allfiles
 }
 (HERE/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"result":"PASS_CPU_CGLS_GENERATION","output_manifest":str(HERE/"manifest.json"),
  "manifest_sha256":sha(HERE/"manifest.json"),"report_sha256":sha(out),"matrix_shape":A.shape,"nnz":int(A.nnz),
  "iterations":{prec:{k:({q:v for q,v in val.items() if q in ["status","predicted_residual_ratio_to_b","predicted_free_csr_residual_l2_Ns","g_dot_p","descent","exported_direction_l2_m_per_s","vs_fp64_exported_direction_relative_l2","vs_emulated_fp32_exported_direction_relative_l2"]}) for k,val in d.items()} for prec,d in states.items()}},indent=2))
if __name__=="__main__":main()
