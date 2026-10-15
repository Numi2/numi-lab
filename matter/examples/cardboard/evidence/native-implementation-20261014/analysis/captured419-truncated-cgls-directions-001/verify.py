#!/usr/bin/env python3
"""Independently recheck captured419 CGLS payloads and exact CSR metrics."""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,sys,struct
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["VECLIB_MAXIMUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy import sparse
B=Path("/Users/home/cardboard-implementation-20261014")
HERE=Path(__file__).resolve().parent
PKG=B/"analysis/captured419-regularized-directions-001"
CAP=B/"solver/two-stage-first-failure-rich-fd-419/operator-capture"
CAP_PKG=B/"analysis/captured-base-state-cauchy-419-001"
BASE=PKG/"base-replay-input.bin"
ASSEMBLY=B/"snapshots/source-status5-vtccd-diagnostics-125/matter/tools/cardboard_cached_operator_audit.py"
HEADER=struct.Struct("<8s6IQfI")
TARGETS=(8,16,32,64,128)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def norm(x):
 return math.sqrt(max(0.0,math.fsum(float(v)*float(v) for v in np.asarray(x,dtype=np.float64).reshape(-1))))
def dot(a,b):
 return math.fsum(float(x)*float(y) for x,y in zip(np.asarray(a,dtype=np.float64).reshape(-1),np.asarray(b,dtype=np.float64).reshape(-1)))
def req(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 man=json.loads((HERE/"manifest.json").read_text())
 for rel,item in man["files"].items():
  p=HERE/rel
  req(p.is_file() and p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],"package artifact mismatch "+rel)
 report=json.loads((HERE/"report.json").read_text())
 req(sha(PKG/"manifest.json")==report["source_bindings"]["captured419_proposal_manifest_sha256"],"proposal manifest changed")
 req(sha(BASE)==report["source_bindings"]["base_replay_input_sha256"],"base replay payload changed")
 req(sha(CAP)==report["source_bindings"]["capture_sha256"],"capture changed")
 req(sha(CAP_PKG/"A_free_csr.npz")==report["source_bindings"]["captured_A_free_csr_sha256"],"CSR changed")
 req(sha(CAP_PKG/"base_rhs.float64le.bin")==report["source_bindings"]["captured_b_sha256"],"RHS changed")
 req(sha(ASSEMBLY)==report["source_bindings"]["source125_assembly_sha256"],"assembly module changed")
 base=BASE.read_bytes()
 magic,version,E,N,T,S,Q,fingerprint,dt,reserved=HEADER.unpack_from(base)
 off=HEADER.size+E*N*80; dbytes=E*N*16
 req((magic,version,E,N,T,S,Q)==(b"NMFDSWP1",1,2,882,1920,15,59),"payload header changed")
 req(len(base)==2195272 and off==141168 and dbytes==28224,"payload layout changed")
 spec=importlib.util.spec_from_file_location("source125_cgls_verify",ASSEMBLY)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 h,nodes,tets,integrity=mod.capture_records(CAP)
 _,_,A2,_,metadata=mod.validate_and_assemble(h,nodes,tets,integrity,0)
 A=sparse.load_npz(CAP_PKG/"A_free_csr.npz").tocsr()
 d=(A-A2).tocsr();d.eliminate_zeros()
 req(d.nnz==0 and A.shape==(2538,2538),"source125 CSR reconstruction differs")
 b=np.fromfile(CAP_PKG/"base_rhs.float64le.bin",dtype="<f8")
 free=np.asarray(metadata["free_nodes"],dtype=np.int64)
 req(len(free)==846 and len(b)==2538,"free map/RHS dimensions changed")
 bnorm=norm(b);g=np.asarray(A.T@b).reshape(-1)
 fixed=np.setdiff1d(np.arange(N),free)
 checked=0
 for precision in ("fp64","emulated_fp32"):
  for target in TARGETS:
   row=report["runs"]["outputs"][precision][str(target)]
   req(row["status"]=="exported",f"missing {precision}/{target}")
   dp=HERE/row["direction_file"];pp=HERE/row["payload_file"]
   direction=dp.read_bytes();payload=pp.read_bytes()
   req(sha(dp)==row["direction_sha256"] and sha(pp)==row["payload_sha256"],"direction/payload hash mismatch")
   req(payload[:off]==base[:off] and payload[off+dbytes:]==base[off+dbytes:],"non-direction bytes changed")
   req(payload[off:off+dbytes]==direction,"embedded direction mismatch")
   full=np.frombuffer(direction,dtype="<f4").reshape(E,N,4)
   req(not np.any(full[1]) and not np.any(full[:,:,3]) and not np.any(full[0,fixed,:3]),"invalid quiet/fixed/w direction entries")
   x=np.asarray(full[0,free,:3],dtype=np.float64).reshape(-1)
   residual=b-np.asarray(A@x).reshape(-1)
   rp=norm(residual); gp=dot(g,x); normal=norm(np.asarray(A.T@residual).reshape(-1))
   values={
    "exported_direction_l2_m_per_s":norm(x),
    "exported_direction_linf_m_per_s":float(np.max(np.abs(x))),
    "predicted_free_csr_residual_l2_Ns":rp,
    "predicted_free_csr_residual_linf_Ns":float(np.max(np.abs(residual))),
    "predicted_residual_ratio_to_b":rp/bnorm,
    "merit_fraction_drop":1-(rp/bnorm)**2,
    "g_dot_p":gp,
    "exact_Atranspose_residual_l2":normal}
   for key,val in values.items():
    expected=float(row[key]);req(math.isclose(val,expected,rel_tol=1e-11,abs_tol=1e-25),f"{precision}/{target} {key} mismatch {val} vs {expected}")
   req(gp>0,f"{precision}/{target} not a descent direction")
   checked+=1
 # Check precision-comparison metrics are measured from the actual exported payloads.
 for target in TARGETS:
  a=report["runs"]["outputs"]["fp64"][str(target)]
  c=report["runs"]["outputs"]["emulated_fp32"][str(target)]
  xa=np.fromfile(HERE/a["direction_file"],dtype="<f4").reshape(E,N,4)[0,free,:3].astype(np.float64).reshape(-1)
  xc=np.fromfile(HERE/c["direction_file"],dtype="<f4").reshape(E,N,4)[0,free,:3].astype(np.float64).reshape(-1)
  rel=norm(xa-xc)/max(norm(xa),1e-300)
  req(math.isclose(rel,float(a["vs_emulated_fp32_exported_direction_relative_l2"]),rel_tol=1e-12,abs_tol=1e-20),"FP64/FP32 stability metric mismatch")
 print(json.dumps({"result":"PASS_CGLS_EXPORTED_PAYLOADS_AND_EXACT_CSR_RECOMPUTE","directions":checked,"matrix_shape":list(A.shape),"csr_nnz":int(A.nnz),"physical_transactions":0,"native_or_gpu_run":False},indent=2))
if __name__=="__main__":main()
