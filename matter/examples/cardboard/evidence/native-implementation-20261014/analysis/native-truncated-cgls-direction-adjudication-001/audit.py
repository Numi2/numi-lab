#!/usr/bin/env python3
"""Source-bound CPU adjudication of native CGLS directions 440.

Checks native source125 action/residual replay against the exact captured CSR,
center parity against native424, every emitted artifact hash, and profile/status
bindings. This is not an accepted Newton step, nonlinear solve certificate, or
physical transaction.
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
NATIVE=B/"analysis/native-truncated-cgls-direction-replay-440"
PROPOSAL=B/"analysis/captured419-truncated-cgls-directions-001"
HELPER=B/"analysis/native-residual-direction-replay-419-001"
CAPTURE_PKG=B/"analysis/captured-base-state-cauchy-419-001"
CAPTURE=B/"solver/two-stage-first-failure-rich-fd-419/operator-capture"
ASSEMBLY=B/"snapshots/source-status5-vtccd-diagnostics-125/matter/tools/cardboard_cached_operator_audit.py"
SOURCE_BUILD=Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json")
PACK=B/"solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack"
BASE424=B/"analysis/native-residual-fd-sweep-result-424/result"
PROFILES={"fp64_iter032":("fp64","32"),"fp64_iter064":("fp64","64"),"fp64_iter128":("fp64","128"),"emulated_fp32_iter016":("emulated_fp32","16"),"emulated_fp32_iter128":("emulated_fp32","128")}
ALPHAS=(1.0,0.5,0.25,0.125,0.0625,0.03125)
HEADER=struct.Struct("<8s6IQfI")

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
def nrm(x)->float:
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    return math.sqrt(max(0.0,math.fsum(float(v)*float(v) for v in a)))
def maxabs(x)->float:
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(a))) if a.size else 0.0
def require(ok:bool,msg:str):
    if not ok: raise AssertionError(msg)
def hash_inventory(root:Path, inventory:dict, label:str)->int:
    for rel,meta in inventory.items():
        p=root/rel; require(p.is_file(),f"{label}: missing {rel}")
        if isinstance(meta,dict): expected=meta["sha256"]; size=meta.get("bytes")
        else: expected=meta; size=None
        require(sha(p)==expected,f"{label}: SHA mismatch {rel}")
        if size is not None: require(p.stat().st_size==size,f"{label}: size mismatch {rel}")
    return len(inventory)
def main():
    native_manifest=json.loads((NATIVE/"manifest.json").read_text())
    native_count=hash_inventory(NATIVE,native_manifest["files"],"native440")
    require(native_count==278,f"native440 inventory count changed: {native_count}")
    receipt=json.loads((NATIVE/"receipt.json").read_text())
    require(receipt.get("physical_transactions")==0,"native440 reports physical transactions")
    require(receipt.get("prepared_inputs_unchanged") is True,"prepared inputs changed")
    require(receipt.get("helper_and_build_bindings_unchanged") is True,"helper/build binding not retained")
    require(all(v.get("exit_code")==0 and not v.get("timed_out") for v in receipt["commands"].values()),"failed/timed-out native command")
    require(len(receipt["commands"])==10,"expected five preflights and five native commands")
    pred=json.loads((NATIVE/"prediction.json").read_text())
    proposal_manifest_path=PROPOSAL/"manifest.json"
    require(sha(proposal_manifest_path)==pred["package_manifest_sha256"],"CGLS package manifest hash mismatch")
    proposal_manifest=json.loads(proposal_manifest_path.read_text())
    proposal_count=hash_inventory(PROPOSAL,proposal_manifest["files"],"CGLS package")
    proposal_report_path=PROPOSAL/"report.json"
    require(sha(proposal_report_path)==proposal_manifest["report_sha256"],"CGLS report hash mismatch")
    proposal_report=json.loads(proposal_report_path.read_text())
    helper_program_path=HELPER/"compiled-program.json"
    require(sha(helper_program_path)==pred["helper_program_sha256"],"compiled helper program binding mismatch")
    program=json.loads(helper_program_path.read_text())
    helper_manifest=json.loads((HELPER/"manifest.json").read_text())
    require(helper_manifest["source125_build_sha256"]==pred["source_build_sha256"],"helper source build identity mismatch")
    require(program["base_manifest_sha256"]==helper_manifest["base_sweep_manifest_sha256"],"helper base manifest mismatch")
    require(proposal_report["source_bindings"]["captured419_proposal_manifest_sha256"]==proposal_manifest["input_manifest_sha256"],"CGLS input proposal manifest mismatch")
    helper_count=hash_inventory(HELPER,program["files"],"native helper program map")
    source_build=json.loads(SOURCE_BUILD.read_text())
    require(sha(SOURCE_BUILD)==pred["source_build_sha256"],"source125 build manifest SHA mismatch")
    require(pred["source_build_sha256"]=="35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6","source125 build identity mismatch")
    build_checked=0
    for raw,expected in source_build["artifacts_sha256"].items():
        p=Path(raw); require(p.is_file() and sha(p)==expected,f"source125 artifact mismatch: {raw}"); build_checked+=1
    # Verify exact helper source and executable entries, in addition to the map inventory.
    helper_bin=HELPER/"build/native_residual_direction"
    helper_source=HELPER/"native_residual_direction.mm"
    require(program["files"].get("native_residual_direction.mm")==sha(helper_source),"helper source map mismatch")
    require(program["files"].get("build/native_residual_direction")==sha(helper_bin),"helper binary map mismatch")
    for name,(mode,it) in PROFILES.items():
        cmd=pred["commands"][name]
        require(Path(cmd[0])==helper_bin and Path(cmd[1])==PACK and Path(cmd[3])==NATIVE/name,f"{name}: command binding mismatch")
        require(sha(PACK)==proposal_report["source_bindings"]["compiled_package_sha256"],"compiled package hash mismatch")
        require(sha(CAPTURE)==proposal_report["source_bindings"]["capture_sha256"],"source capture mismatch")
        require(sha(ASSEMBLY)==proposal_report["source_bindings"]["source125_assembly_sha256"],"assembly source mismatch")
    # Exact source125 CSR reconstruction and free-node indexing.
    spec=importlib.util.spec_from_file_location("source125_assembly_440",ASSEMBLY)
    require(spec is not None and spec.loader is not None,"cannot load source125 assembly module")
    assembly=importlib.util.module_from_spec(spec); sys.modules[spec.name]=assembly; spec.loader.exec_module(assembly)
    h,node_rows,tet_rows,integrity=assembly.capture_records(CAPTURE)
    _,_,reconstructed,_,metadata=assembly.validate_and_assemble(h,node_rows,tet_rows,integrity,0)
    A=sparse.load_npz(CAPTURE_PKG/"A_free_csr.npz").tocsr()
    delta=(A-reconstructed).tocsr(); delta.eliminate_zeros()
    require(delta.nnz==0 and A.shape==(2538,2538),"captured CSR differs from exact source assembly")
    require(int(A.nnz)==71046,"CSR sparsity changed")
    free=np.asarray(metadata["free_nodes"],dtype=np.int64); require(free.size==846,"free node map changed")
    b=np.fromfile(CAPTURE_PKG/"base_rhs.float64le.bin",dtype="<f8"); require(b.size==2538,"RHS shape changed")
    base_norm=nrm(b)
    old_root=BASE424.parent; old_manifest=json.loads((old_root/"manifest.json").read_text())
    old_checked=hash_inventory(old_root,old_manifest.get("files",{}),"native424")
    per={}
    output_index=proposal_report["runs"]["outputs"]
    for name,(mode,it) in PROFILES.items():
        data=output_index[mode][it]; input_file=PROPOSAL/data["payload_file"]
        require(sha(input_file)==data["payload_sha256"],f"{name}: payload hash mismatch")
        payload=input_file.read_bytes()
        magic,version,envs,nodes,tets,stride,params,fingerprint,dt,reserved=HEADER.unpack_from(payload)
        require((magic,version,envs,nodes,tets,stride,params)==(b"NMFDSWP1",1,2,882,1920,15,59),f"{name}: payload header mismatch")
        require(len(payload)==2195272 and len(data["direction_file"])>0,f"{name}: payload size/metadata mismatch")
        direction_file=PROPOSAL/data["direction_file"]
        direction_raw=direction_file.read_bytes()
        require(sha(direction_file)==data["direction_sha256"],f"{name}: direction file hash mismatch")
        offset=141168; size=2*882*16
        require(payload[offset:offset+size]==direction_raw,f"{name}: embedded direction differs from package direction")
        full=np.frombuffer(direction_raw,dtype="<f4").reshape(2,882,4)
        require(np.all(full[1]==0.0) and np.all(full[:,:,3]==0.0),f"{name}: quiet env or w lane nonzero")
        require(np.all(full[0,np.setdiff1d(np.arange(882),free),:3]==0.0),f"{name}: fixed node direction nonzero")
        p=full[0,free,:3].astype(np.float64).reshape(-1); Ap=np.asarray(A@p).reshape(-1)
        require(data.get("descent") is True and float(data["g_dot_p"])>0.0,"CGLS profile is not a checked descent direction")
        sub=NATIVE/name
        cached=np.fromfile(sub/"cached-action.float4.bin",dtype="<f4").reshape(2,882,4)[0,free,:3].astype(np.float64).reshape(-1)
        fresh=np.fromfile(sub/"fresh-action.float4.bin",dtype="<f4").reshape(2,882,4)[0,free,:3].astype(np.float64).reshape(-1)
        mass=np.fromfile(sub/"mass-only-action.float4.bin",dtype="<f4").reshape(2,882,4)[0,free,:3].astype(np.float64).reshape(-1)
        center_file_hashes={}
        for filename in ("sample-0-0-nodes.bin","sample-0-0-residual.bin","sample-0-0-forces.bin"):
            q=(sub/filename).read_bytes(); require(q==(BASE424/filename).read_bytes(),f"{name}: center differs from 424: {filename}")
            center_file_hashes[filename]=sha(sub/filename)
        center=np.fromfile(sub/"sample-0-0-residual.bin",dtype="<f4").reshape(2,882,4)[0,free,:3].astype(np.float64).reshape(-1)
        require(center.size==b.size,"center residual dimension mismatch")
        sweep=[json.loads(line) for line in (sub/"residual-sweep.jsonl").read_text().splitlines()]
        expected_keys={(0,0),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(4,0),(4,1),(5,0),(5,1),(6,0),(6,1),(7,0),(7,1)}
        require(len(sweep)==15 and {(int(q["sample"]),int(q["side"])) for q in sweep}==expected_keys,f"{name}: sweep sample coverage mismatch")
        curve=[]; actual_positive={}
        for q in sorted(sweep,key=lambda v:(int(v["sample"]),int(v["side"]))):
            k=int(q["sample"]); side=int(q["side"]); eps=float(q["epsilon"])
            require(q["status"]==[0,0] and q["history_unchanged"] is True,f"{name}: trial failed or history changed at {k}/{side}")
            rr=np.fromfile(sub/f"sample-{k}-{side}-residual.bin",dtype="<f4").reshape(2,882,4)[0,free,:3].astype(np.float64).reshape(-1)
            signed=(-1.0 if side else 1.0)*eps
            predicted=b-np.asarray(A@(signed*p)).reshape(-1)
            actual_norm=nrm(rr); pred_norm=nrm(predicted)
            item={"sample":k,"side":side,"epsilon":eps,"signed_alpha":signed,"status":q["status"],"history_unchanged":q["history_unchanged"],
                "native_residual_l2_Ns":actual_norm,"native_residual_ratio_to_center":actual_norm/max(nrm(center),1e-300),
                "native_residual_linf_Ns":maxabs(rr),"exact_csr_linear_prediction_l2_Ns":pred_norm,
                "exact_csr_linear_ratio_to_rhs":pred_norm/max(base_norm,1e-300),"native_minus_linear_l2_Ns":nrm(rr-predicted),
                "native_minus_linear_relative_to_native":nrm(rr-predicted)/max(actual_norm,1e-300)}
            curve.append(item)
            if side==0 and eps>0: actual_positive[eps]=actual_norm
        positive=[1.0,0.5,0.25,0.125,0.0625,0.03125]
        require(abs(nrm(b-Ap)/max(base_norm,1e-300)-float(data["predicted_residual_ratio_to_b"]))<2e-6,f"{name}: CGLS report linear prediction mismatch")
        best=min(positive,key=lambda x:actual_positive[x])
        ordered=[actual_positive[x] for x in positive]
        require(all(ordered[i]<ordered[i+1] for i in range(len(ordered)-1)),f"{name}: native residual did not worsen monotonically as alpha decreased")
        predicted_vals={q["epsilon"]:q["exact_csr_linear_prediction_l2_Ns"] for q in curve if q["side"]==0 and q["epsilon"] in positive}
        pred_best=min(positive,key=lambda x:predicted_vals[x])
        per[name]={"profile":{"mode":mode,"iterations":int(it),"payload_sha256":sha(input_file),"direction_sha256":sha(direction_file),"direction_l2_m_per_s":nrm(p),"direction_linf_m_per_s":maxabs(p),"g_dot_p":data["g_dot_p"],"directional_derivative":data["least_squares_directional_derivative"],"CGLS_exact_CSR_prediction_ratio":data["predicted_residual_ratio_to_b"]},
            "native_actions":{"cached_vs_exact_CSR_l2_Ns":nrm(cached-Ap),"cached_vs_exact_CSR_relative_l2":nrm(cached-Ap)/max(nrm(Ap),1e-300),"fresh_vs_exact_CSR_l2_Ns":nrm(fresh-Ap),"fresh_vs_exact_CSR_relative_l2":nrm(fresh-Ap)/max(nrm(Ap),1e-300),"cached_vs_fresh_l2_Ns":nrm(cached-fresh),"cached_vs_fresh_relative_l2":nrm(cached-fresh)/max(nrm(cached),1e-300),"mass_action_l2_Ns":nrm(mass),"cached_CSR_cosine":float(np.dot(cached,Ap)/max(nrm(cached)*nrm(Ap),1e-300)),"fresh_CSR_cosine":float(np.dot(fresh,Ap)/max(nrm(fresh)*nrm(Ap),1e-300))},
            "center":{"exact_byte_equal_to_native424":True,"file_sha256":center_file_hashes,"native_residual_l2_Ns":nrm(center),"captured_CSR_rhs_l2_Ns":base_norm,"center_minus_rhs_l2_Ns":nrm(center-b),"center_minus_rhs_linf_Ns":maxabs(center-b)},
            "curve":curve,"best_tested_positive_alpha":{"native_alpha":best,"native_residual_l2_Ns":actual_positive[best],"native_residual_ratio_to_center":actual_positive[best]/max(nrm(center),1e-300),"linear_prediction_alpha":pred_best,"linear_prediction_l2_Ns":predicted_vals[pred_best],"linear_prediction_ratio_to_rhs":predicted_vals[pred_best]/max(base_norm,1e-300),"native_best_is_alpha1":best==1.0}}
    report={"schema":"numi.cardboard.native440-truncated-cgls-direction-adjudication.v1","classification":"CPU-only adjudication of source125 native CGLS direction residual/operator replay; zero physical transactions; no nonlinear acceptance or certificate claim.","native440":{"path":str(NATIVE),"prediction_sha256":sha(NATIVE/"prediction.json"),"receipt_sha256":sha(NATIVE/"receipt.json"),"manifest_sha256":sha(NATIVE/"manifest.json"),"files_verified":native_count,"commands_all_exit0":True,"physical_transactions":0},"bindings":{"cgls_package_manifest_sha256":sha(PROPOSAL/"manifest.json"),"cgls_package_files_verified":proposal_count,"cgls_report_sha256":sha(PROPOSAL/"report.json"),"native_helper_compiled_program_sha256":sha(helper_program_path),"native_helper_manifest_sha256":sha(HELPER/"manifest.json"),"helper_files_verified":helper_count,"helper_source_sha256":sha(helper_source),"helper_binary_sha256":sha(helper_bin),"compiled_nmatterpack_sha256":sha(PACK),"source125_build_manifest_sha256":sha(SOURCE_BUILD),"source125_artifacts_verified":build_checked,"source125_capture_sha256":sha(CAPTURE),"source125_assembly_sha256":sha(ASSEMBLY),"csr_sha256":sha(CAPTURE_PKG/"A_free_csr.npz"),"rhs_sha256":sha(CAPTURE_PKG/"base_rhs.float64le.bin"),"csr_reconstructed_exactly":True,"csr_shape":list(A.shape),"csr_nnz":int(A.nnz),"free_node_count":int(free.size),"native424_manifest_sha256":sha(old_root/"manifest.json"),"native424_files_verified":old_checked},"profiles":per,"interpretation":{"center":"All three center outputs are exact byte matches to source-bound native424 for each direction profile. This binds replay to the same captured linearization center, not to an accepted Newton solve.","action":"Native cached and fresh A*p exports are compared to the exact unsymmetric captured source125 CSR on each exported float32 direction. This checks operator replay, not convergence.","residual":"For each signed alpha, native production candidate/projection/force/residual output is compared with b-alpha*A*p. Ratios are against center native residual and exact CSR right-hand side respectively.","best_alpha":"Best alpha is the minimum native residual among sampled positive values 1,1/2,...,1/32. The alpha=2 diagnostic point is excluded. This is not an Armijo acceptance and no physical transaction occurred.","limits":["All native statuses and material-history checks pass, but these are offline replay diagnostics only.","CGLS truncation does not inherit the original Newton certificate; no direction here was installed as a production correction.","The best sampled alpha is alpha=1 for all five profiles; every smaller tested positive alpha gives a larger native residual.","FP64 CGLS128 predicts a lower linear residual than CGLS64 but produces a worse nonlinear alpha=1 residual; do not infer linear improvement transfers to nonlinear descent.","No dynamic contact, full assembly, accepted time step, or physical qualification is tested."]}}
    out=HERE/"report.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"result":"PASS_SOURCE_AND_NATIVE440_ADJUDICATION","native_files_verified":native_count,"proposal_files_verified":proposal_count,"helper_files_verified":helper_count,"source125_artifacts_verified":build_checked,"csr_shape":list(A.shape),"csr_nnz":int(A.nnz),"profiles":{k:{"alpha1_native_ratio":next(q["native_residual_ratio_to_center"] for q in v["curve"] if q["sample"]==1 and q["side"]==0),"best_alpha":v["best_tested_positive_alpha"]["native_alpha"],"best_ratio":v["best_tested_positive_alpha"]["native_residual_ratio_to_center"],"alpha1_linear_ratio":next(q["exact_csr_linear_ratio_to_rhs"] for q in v["curve"] if q["sample"]==1 and q["side"]==0)} for k,v in per.items()},"report_sha256":sha(out)},indent=2))
if __name__=="__main__": main()
