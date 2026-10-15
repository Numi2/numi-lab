#!/usr/bin/env python3
"""Source-bound CPU audit of native regularized-direction replay 436.

This verifies emitted native operator actions and residual curves against the
exact captured source125 CSR and the prior native 424 center. It makes no
physical transaction, nonlinear acceptance, or certificate claim.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, struct, sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy import sparse

B=Path("/Users/home/cardboard-implementation-20261014")
HERE=Path(__file__).resolve().parent
NATIVE=B/"analysis/native-regularized-direction-replay-436"
PROPOSAL=B/"analysis/captured419-regularized-directions-001"
HELPER=B/"analysis/native-residual-direction-replay-419-001"
CAPTURE_PKG=B/"analysis/captured-base-state-cauchy-419-001"
CAPTURE=B/"solver/two-stage-first-failure-rich-fd-419/operator-capture"
ASSEMBLY=B/"snapshots/source-status5-vtccd-diagnostics-125/matter/tools/cardboard_cached_operator_audit.py"
SOURCE_BUILD=Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json")
PACK=B/"solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack"
BASE424=B/"analysis/native-residual-fd-sweep-result-424/result"
HEADER=struct.Struct("<8s6IQfI")
PROFILES=("raw_cauchy","tikhonov_sigma_0.01","tikhonov_sigma_0.001","tikhonov_sigma_0.0001")
ALPHAS=(1.0,0.5,0.25,0.125,0.0625,0.03125)

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

def import_assembly():
    spec=importlib.util.spec_from_file_location("source125_assembly436",ASSEMBLY)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import source125 assembly")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod

def require(condition:bool,message:str):
    if not condition: raise AssertionError(message)

def hash_inventory(root:Path, inventory:dict, label:str):
    checked=0
    for rel,meta in inventory.items():
        path=root/rel
        require(path.is_file(),f"{label}: missing {rel}")
        if isinstance(meta,dict):
            expected=meta["sha256"]; size=meta.get("bytes")
        else:
            expected=meta; size=None
        actual=sha(path)
        require(actual==expected,f"{label}: SHA mismatch {rel}: {actual} != {expected}")
        if size is not None: require(path.stat().st_size==size,f"{label}: size mismatch {rel}")
        checked+=1
    return checked

def main():
    native_manifest=json.loads((NATIVE/"manifest.json").read_text())
    require(len(native_manifest["files"])==223,"native436 inventory count changed")
    native_files=hash_inventory(NATIVE,native_manifest["files"],"native436")
    receipt=json.loads((NATIVE/"receipt.json").read_text())
    require(receipt.get("physical_transactions")==0,"native436 reports physical transactions")
    require(receipt.get("prepared_inputs_unchanged") is True,"native436 inputs not confirmed unchanged")
    require(all(x.get("exit_code")==0 and not x.get("timed_out") for x in receipt["commands"].values()),
            "native436 contains a failed or timed-out preflight/native command")
    require(len(receipt["commands"])==8,"expected four preflights and four native replays")

    pred=json.loads((NATIVE/"prediction.json").read_text())
    proposal_manifest_path=PROPOSAL/"manifest.json"
    require(sha(proposal_manifest_path)==pred["package_manifest_sha256"],"proposal package manifest changed")
    proposal_manifest=json.loads(proposal_manifest_path.read_text())
    proposal_manifest_checked=0
    for item in proposal_manifest["artifacts"]:
        path=PROPOSAL/item["path"]
        require(path.is_file() and path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],
                f"proposal artifact mismatch {item['path']}")
        proposal_manifest_checked+=1
    proposal_report=json.loads((PROPOSAL/"report.json").read_text())
    require(sha(HELPER/"compiled-program.json")==pred["program_binding_sha256"],
            "helper compiled-program binding changed")
    program=json.loads((HELPER/"compiled-program.json").read_text())
    helper_checked=hash_inventory(HELPER,program["files"],"helper compiled-program")
    helper_binary=Path(pred["commands"]["raw_cauchy"][0])
    require(helper_binary==HELPER/"build/native_residual_direction","436 command helper path mismatch")
    require(all(Path(v[0])==helper_binary for v in pred["commands"].values()),
            "not every profile used the same helper executable")
    require(all(Path(v[1])==PACK for v in pred["commands"].values()),
            "not every profile used the same compiled package")
    require(sha(PACK)==proposal_manifest["compiled_package_sha256"],"compiled package hash mismatch")
    require(pred["source_build_sha256"]==proposal_manifest["source125_build_sha256"],
            "source build SHA differs from proposal")
    build=json.loads(SOURCE_BUILD.read_text())
    require(sha(SOURCE_BUILD)==pred["source_build_sha256"],"source125 build manifest changed")
    protected={Path(k):v for k,v in build["artifacts_sha256"].items()}
    protected[ SOURCE_BUILD ]=sha(SOURCE_BUILD)
    build_artifacts_checked=0
    for path,expected in protected.items():
        require(path.is_file() and sha(path)==expected,f"source125 artifact changed: {path}")
        build_artifacts_checked+=1
    # Explicitly audit the helper mapping omitted from runner.py's protected build list.
    helper_source=HELPER/"native_residual_direction.mm"
    helper_map=program["files"]
    require(helper_map["native_residual_direction.mm"]==sha(helper_source),"helper source mapping mismatch")
    require(helper_map["build/native_residual_direction"]==sha(helper_binary),"helper binary mapping mismatch")
    require(sha(HELPER/"manifest.json")==proposal_manifest["source_input_bindings"]["native_direction_manifest"]["sha256"],
            "source helper manifest changed")
    require(program["base_manifest_sha256"]==proposal_report["source_capture"]["base_replay_manifest_sha256"],
            "helper base sweep manifest mismatch")
    require(sha(CAPTURE)==proposal_manifest["capture_sha256"],"source125 capture changed")
    require(sha(ASSEMBLY)==proposal_manifest["source_input_bindings"]["source125_assembly_module"]["sha256"],
            "source125 assembly module changed")
    require(sha(PACK)==proposal_manifest["source_input_bindings"]["compiled_package"]["sha256"],
            "compiled package proposal binding mismatch")

    # Verify the 424 files supporting exact center parity, not only their current bytes.
    old_root=BASE424.parent
    old_manifest=json.loads((old_root/"manifest.json").read_text())
    old_file_inventory=old_manifest.get("files",{})
    old_files_checked=hash_inventory(old_root,old_file_inventory,"native424") if old_file_inventory else 0

    # Verify proposal package and retrieve exact directions and captured source matrix/RHS.
    coa=import_assembly()
    header,node_rows,tet_rows,integrity=coa.capture_records(CAPTURE)
    _,_,reconstructed,_,metadata=coa.validate_and_assemble(header,node_rows,tet_rows,integrity,0)
    A=sparse.load_npz(CAPTURE_PKG/"A_free_csr.npz").tocsr()
    diff=(A-reconstructed).tocsr(); diff.eliminate_zeros()
    require(diff.nnz==0 and A.shape==(2538,2538),"CSR replay matrix differs from source125 reconstruction")
    free=np.asarray(metadata["free_nodes"],dtype=np.int64)
    require(len(free)==846,"free-node mapping changed")
    b=np.fromfile(CAPTURE_PKG/"base_rhs.float64le.bin",dtype="<f8")
    require(b.size==2538,"base RHS shape changed")
    base_norm=nrm(b)
    pred_results=proposal_report["results"]
    input_bindings={}
    for profile in PROFILES:
        expected=pred_results[profile]
        ip=PROPOSAL/expected["replay_payload_file"]
        ib=ip.read_bytes()
        require(sha(ip)==expected["replay_payload_sha256"],f"{profile} replay input SHA mismatch")
        magic,version,envs,nodes,tets,stride,param_count,fingerprint,dt,reserved=HEADER.unpack_from(ib)
        require((magic,version,envs,nodes,tets,stride,param_count)==(b"NMFDSWP1",1,2,882,1920,15,59),
                f"{profile} payload header mismatch")
        doff=int(proposal_manifest["payload_format"]["direction_offset_bytes"])
        dbytes=int(proposal_manifest["payload_format"]["direction_bytes"])
        require(len(ib)==2195272 and dbytes==2*882*16,"payload layout or length mismatch")
        direction_bytes=ib[doff:doff+dbytes]
        direction_file=PROPOSAL/expected["direction_file"]
        require(direction_bytes==direction_file.read_bytes(),f"{profile} embedded direction mismatch")
        full=np.frombuffer(direction_bytes,dtype="<f4").reshape(2,882,4)
        require(np.all(full[1]==0.0) and np.all(full[:,:,3]==0.0),f"{profile} quiet/w lanes not zero")
        x=np.asarray(full[0,free,:3],dtype=np.float64).reshape(-1)
        fixed=np.setdiff1d(np.arange(nodes),free)
        require(np.all(full[0,fixed,:3]==0.0),f"{profile} fixed-node direction nonzero")
        Ap=np.asarray(A@x).reshape(-1)
        input_bindings[profile]={"payload_sha256":sha(ip),"direction_sha256":sha(direction_file),
            "header":{"world_fingerprint_hex":f"{fingerprint:016x}","dt_seconds":float(dt)},
            "correction_l2_m_per_s":nrm(x),"csr_action_l2_Ns":nrm(Ap)}
        # Native cached/fresh operator action against exact unsymmetric captured CSR.
        sub=NATIVE/profile
        cached=np.fromfile(sub/"cached-action.float4.bin",dtype="<f4").reshape(2,882,4)
        fresh=np.fromfile(sub/"fresh-action.float4.bin",dtype="<f4").reshape(2,882,4)
        mass=np.fromfile(sub/"mass-only-action.float4.bin",dtype="<f4").reshape(2,882,4)
        cached_free=np.asarray(cached[0,free,:3],dtype=np.float64).reshape(-1)
        fresh_free=np.asarray(fresh[0,free,:3],dtype=np.float64).reshape(-1)
        mass_free=np.asarray(mass[0,free,:3],dtype=np.float64).reshape(-1)
        action_delta=cached_free-fresh_free
        cached_error=cached_free-Ap
        fresh_error=fresh_free-Ap
        input_bindings[profile]["native_actions"]={
            "cached_vs_exact_csr_abs_l2_Ns":nrm(cached_error),
            "cached_vs_exact_csr_rel_l2":nrm(cached_error)/max(nrm(Ap),1e-300),
            "fresh_vs_exact_csr_abs_l2_Ns":nrm(fresh_error),
            "fresh_vs_exact_csr_rel_l2":nrm(fresh_error)/max(nrm(Ap),1e-300),
            "cached_vs_fresh_abs_l2_Ns":nrm(action_delta),
            "cached_vs_fresh_rel_l2":nrm(action_delta)/max(nrm(cached_free),1e-300),
            "mass_only_action_l2_Ns":nrm(mass_free),
            "cached_csr_cosine":float(np.dot(cached_free,Ap)/(max(nrm(cached_free)*nrm(Ap),1e-300))),
            "fresh_csr_cosine":float(np.dot(fresh_free,Ap)/(max(nrm(fresh_free)*nrm(Ap),1e-300)))}
        # Exact center parity to both the prior 424 native center and all other profiles.
        center_hashes={}
        center_data={}
        for filename in ("sample-0-0-nodes.bin","sample-0-0-residual.bin","sample-0-0-forces.bin"):
            current=(sub/filename).read_bytes()
            old=(BASE424/filename).read_bytes()
            require(current==old,f"{profile} center output differs from source-bound native424: {filename}")
            center_hashes[filename]=sha(sub/filename)
            center_data[filename]=current
        input_bindings[profile]["center_files"]={
            "exact_to_424":True,"sha256":center_hashes}
        r0_all=np.frombuffer(center_data["sample-0-0-residual.bin"],dtype="<f4").reshape(2,882,4)
        r0=np.asarray(r0_all[0,free,:3],dtype=np.float64).reshape(-1)
        input_bindings[profile]["center_residual_vs_exact_b"]={
            "native_center_l2_Ns":nrm(r0),
            "exact_b_l2_Ns":base_norm,
            "difference_l2_Ns":nrm(r0-b),
            "difference_linf_Ns":maxabs(r0-b)}
        # Actual nonlinear native residual path and its exact linear CSR prediction.
        lines=[json.loads(line) for line in (sub/"residual-sweep.jsonl").read_text().splitlines()]
        require(len(lines)==15,f"{profile} sweep has {len(lines)} records rather than 15")
        expected_keys={(0,0),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1),(4,0),(4,1),(5,0),(5,1),(6,0),(6,1),(7,0),(7,1)}
        require({(int(q["sample"]),int(q["side"])) for q in lines}==expected_keys,
                f"{profile} sweep sample/side set changed")
        curve=[]
        actual_by_alpha={}
        predicted_by_alpha={}
        for row in sorted(lines,key=lambda q:(int(q["sample"]),int(q["side"]))):
            k=int(row["sample"]); side=int(row["side"]); eps=float(row["epsilon"])
            require(row["status"]==[0,0] and row["history_unchanged"] is True,
                    f"{profile} non-success trial/status/history at k={k},side={side}")
            rb=np.fromfile(sub/f"sample-{k}-{side}-residual.bin",dtype="<f4").reshape(2,882,4)
            rr=np.asarray(rb[0,free,:3],dtype=np.float64).reshape(-1)
            sign=-1.0 if side else 1.0
            linear=b-np.asarray(A@(sign*eps*x)).reshape(-1)
            actual=nrm(rr)
            lin=nrm(linear)
            error=nrm(rr-linear)
            record={"sample":k,"side":side,"alpha":eps,"signed_alpha":sign*eps,
                "status":[int(v) for v in row["status"]],"history_unchanged":row["history_unchanged"],
                "native_nonlinear_residual_l2_Ns":actual,
                "exact_csr_linear_prediction_l2_Ns":lin,
                "native_minus_linear_vector_l2_Ns":error,
                "native_residual_ratio_to_native_center":actual/max(nrm(r0),1e-300),
                "linear_prediction_ratio_to_exact_b":lin/max(base_norm,1e-300),
                "native_residual_linf_Ns":maxabs(rr),
                "linear_prediction_linf_Ns":maxabs(linear)}
            curve.append(record)
            if side==0 and eps>0:
                actual_by_alpha[eps]=actual
                predicted_by_alpha[eps]=lin
        tested=[a for a in ALPHAS]
        actual_best=min(tested,key=lambda a:actual_by_alpha[a])
        predicted_best=min(tested,key=lambda a:predicted_by_alpha[a])
        input_bindings[profile]["residual_curve"]=curve
        input_bindings[profile]["best_tested_backtracking_alpha"]={
            "tested_alphas_excluding_non_armijo_alpha2":list(tested),
            "native_nonlinear_best_alpha":actual_best,
            "native_nonlinear_best_l2_Ns":actual_by_alpha[actual_best],
            "native_nonlinear_best_ratio_to_center":actual_by_alpha[actual_best]/max(nrm(r0),1e-300),
            "exact_csr_linear_best_alpha":predicted_best,
            "exact_csr_linear_best_l2_Ns":predicted_by_alpha[predicted_best],
            "exact_csr_linear_best_ratio_to_b":predicted_by_alpha[predicted_best]/max(base_norm,1e-300),
            "native_and_linear_best_alpha_match":actual_best==predicted_best}
    report={
        "schema":"numi.cardboard.native436-regularized-direction-adjudication.v1",
        "classification":"CPU adjudication of source125 native residual/operator replay; no physical transaction, no nonlinear acceptance, and no certificate claim",
        "native436":{"path":str(NATIVE),"receipt_sha256":sha(NATIVE/"receipt.json"),
            "manifest_sha256":sha(NATIVE/"manifest.json"),"inventory_files_verified":native_files,
            "commands_all_exit0":True,"physical_transactions":0},
        "source_bindings":{
            "source125_build_manifest":str(SOURCE_BUILD),"source125_build_manifest_sha256":sha(SOURCE_BUILD),
            "source125_artifacts_verified":build_artifacts_checked,
            "compiled_program":str(HELPER/"compiled-program.json"),
            "compiled_program_sha256":sha(HELPER/"compiled-program.json"),
            "compiled_program_file_map_verified":helper_checked,
            "helper_source_sha256":sha(helper_source),"helper_binary_sha256":sha(helper_binary),
            "helper_compile_argv":program["compile_argv"],
            "helper_manifest_sha256":sha(HELPER/"manifest.json"),
            "compiled_nmatterpack_sha256":sha(PACK),
            "capture_sha256":sha(CAPTURE),"csr_sha256":sha(CAPTURE_PKG/"A_free_csr.npz"),
            "base_rhs_sha256":sha(CAPTURE_PKG/"base_rhs.float64le.bin"),
            "proposal_manifest_sha256":sha(proposal_manifest_path),
            "proposal_artifacts_verified":proposal_manifest_checked,
            "native424_files_verified":old_files_checked,
            "native424_center_manifest_sha256":sha(old_root/"manifest.json"),
            "source125_assembly_sha256":sha(ASSEMBLY),
            "csr_reconstruction_exact":True,"csr_shape":list(A.shape),"csr_nnz":int(A.nnz),
            "free_node_count":int(len(free))},
        "profiles":input_bindings,
        "interpretation":{
            "linear_curve":"The CSR curve is b - alpha*A*p, with signed alpha for each measured side; it is a local linear prediction at the captured source125 center, not an Armijo acceptance or physical transaction.",
            "alpha1":"The native alpha=1 residual is an independently evaluated nonlinear residual at the exact trial from the production candidate/geometry/material-force/residual kernels; all reported sweep statuses are successful and accepted material history is byte-preserved.",
            "best_alpha":"The reported best tested alpha is only the minimum residual norm among the sampled positive backtracking values 1,1/2,...,1/32. alpha=2 is reported in the curve but excluded from this Armijo-like candidate set.",
            "center":"All three center files (candidate nodes, residual, internal element forces) are byte-identical across four profiles and to native424. This verifies replay center parity, not direction validity as an accepted Newton solve.",
            "actions":"Cached and uncached native operator actions are compared with exact captured unsymmetrized CSR action on the exported FP32 vector. CSR agreement is a captured-operator consistency check; it does not certify nonlinear descent or terminal convergence.",
            "limitations":["The run does not execute physical transactions or perform CCD/line-search acceptance.","The original Newton certificate applies only to its original captured correction, not any replacement direction.","Nonlinear residual evaluation at tested alpha is diagnostic only; even a lower residual is not an accepted physics step or proof of solver convergence.","No noise-floor or physical qualification claim is made.","Fixed source125/no-contact explicit FEM scope only; no cohesive, rigid, field, or other coupled rows."]
        }
    }
    out=HERE/"report.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"result":"PASS_SOURCE_AND_NATIVE_REPLAY_ADJUDICATION",
        "native436_file_count":native_files,"source125_artifacts_verified":build_artifacts_checked,
        "helper_files_verified":helper_checked,"native424_files_verified":old_files_checked,
        "csr_shape":list(A.shape),"csr_nnz":int(A.nnz),
        "profiles":{p:{
            "native_cached_rel_action_error":report["profiles"][p]["native_actions"]["cached_vs_exact_csr_rel_l2"],
            "native_fresh_rel_action_error":report["profiles"][p]["native_actions"]["fresh_vs_exact_csr_rel_l2"],
            "alpha1_actual":next(x for x in report["profiles"][p]["residual_curve"] if x["alpha"]==1 and x["side"]==0)["native_nonlinear_residual_l2_Ns"],
            "alpha1_linear":next(x for x in report["profiles"][p]["residual_curve"] if x["alpha"]==1 and x["side"]==0)["exact_csr_linear_prediction_l2_Ns"],
            "native_best_alpha":report["profiles"][p]["best_tested_backtracking_alpha"]["native_nonlinear_best_alpha"],
            "linear_best_alpha":report["profiles"][p]["best_tested_backtracking_alpha"]["exact_csr_linear_best_alpha"]
            } for p in PROFILES},
        "report_sha256":sha(out)},indent=2))
if __name__=="__main__": main()
