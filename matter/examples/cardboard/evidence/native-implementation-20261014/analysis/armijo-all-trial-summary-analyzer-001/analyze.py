#!/usr/bin/env python3
"""Strict adjudication for all-trial Armijo summary capture and physical parity."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from pathlib import Path
from typing import Any

ROOT = Path("/Users/home/cardboard-implementation-20261014")
NATIVE = ROOT / "native"
RUN395 = NATIVE / "full-contact-padded-trial-summary-395"
RUN393 = NATIVE / "full-contact-all-trial-summary-393"
RUN391 = NATIVE / "dimensionless-policy-off-full-contact-391"
STAGES = ["geometry_validation","bulk_material_mechanical","cohesive_residual","private_rigid_contact","deformable_sweep_ccd","deformable_self_narrowphase","free_body_full_residual","measured_predecision"]
STATUS_NAMES = {0:"success",1:"invalid_dispatch",2:"capacity_overflow",3:"nonfinite_input",4:"nonfinite_result",5:"invalid_deformation",6:"contact_failure",7:"linear_solver_failure",8:"unsupported",9:"rigid_world_failure",10:"nonlinear_solver_failure",11:"multiphysics_failure",12:"topology_failure",13:"learned_material_failure",14:"local_material_failure",15:"topology_growth_required"}
DTYPE_SIZE = {"record:NMMatterStatusGPU":48,"float32x4":16,"float32":4}

class AuditError(RuntimeError): pass

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def strict_json_text(s: str, label: str="JSON") -> Any:
    def reject(token: str): raise AuditError(f"{label}: non-standard JSON token {token}")
    try: return json.loads(s,parse_constant=reject)
    except AuditError: raise
    except Exception as e: raise AuditError(f"{label}: invalid JSON: {e}") from e

def strict_json_file(p: Path) -> Any: return strict_json_text(p.read_text(encoding="utf-8"),str(p))

def repair_manifest(raw: str) -> tuple[str,dict[str,Any]]:
    # The native writer emitted one extra brace after phase_order. Keep raw untouched.
    marker = ']},\"payload\":'
    if raw.count(marker)!=1: raise AuditError(f"expected one malformed separator, got {raw.count(marker)}")
    fixed=raw.replace(marker,'],\"payload\":',1)
    if len(fixed.encode())+1!=len(raw.encode()): raise AuditError("repair was not one-byte deletion")
    strict_json_text(fixed,"repaired manifest")
    return fixed,{"raw_bytes":len(raw.encode()),"raw_sha256":sha_bytes(raw.encode()),
        "raw_json_status":"invalid_extra_data_preserved",
        "repair":"delete one extra closing brace before payload",
        "repaired_bytes":len(fixed.encode()),"repaired_sha256":sha_bytes(fixed.encode()),
        "adjudicated_json_status":"strict_json_valid"}

def jnum(x: float) -> float|str:
    if math.isfinite(x): return x
    if math.isnan(x): return "NaN"
    return "+Infinity" if x>0 else "-Infinity"

def decode_status(raw: bytes) -> dict[str,Any]:
    if len(raw)!=48: raise AuditError(f"status record expected 48 bytes, got {len(raw)}")
    u=struct.unpack("<8I",raw[:32]); f=struct.unpack("<4f",raw[32:])
    return {"code":u[0],"code_name":STATUS_NAMES.get(u[0],"unknown"),"environment":u[1],
      "object_index":u[2],"failing_index":u[3],"completed_microsteps":u[4],
      "fgmres_iterations":u[5],"contact_count":u[6],"event_count":u[7],
      "diagnostics":[jnum(float(x)) for x in f],
      "diagnostic_f32_bits":[f"0x{struct.unpack_from('<I',raw,32+4*i)[0]:08x}" for i in range(4)]}

def verify_payload(run: Path, fixed_text: str) -> dict[str,Any]:
    m=strict_json_text(fixed_text,"adjudicated summary")
    if m.get("schema")!="numi.matter.coupled_armijo_trial_summary.v1" or m.get("byte_order")!="little_endian":
        raise AuditError("unexpected summary schema/byte order")
    arrays=m.get("arrays"); budget=int(m.get("line_search_budget",-1)); envs=int(m.get("environment_count",-1))
    if not isinstance(arrays,list) or budget<1 or envs<1: raise AuditError("bad array list/budget/environment count")
    if int(m.get("array_count",-1))!=len(arrays) or len(arrays)>192 or len(arrays)>int(m.get("max_arrays",0)):
        raise AuditError("array count cap/metadata mismatch")
    pref=m.get("payload",{}).get("file")
    if not isinstance(pref,str) or Path(pref).name!="trial-summary.summary.bin": raise AuditError("unexpected payload reference")
    payload=run/"trial-summary.summary.bin" # always read manifest-adjacent bytes
    if payload.is_symlink() or not payload.is_file(): raise AuditError("adjacent payload missing or non-regular")
    blob=payload.read_bytes()
    if len(blob)>32*1024*1024 or len(blob)!=int(m["payload"].get("byte_count",-1)): raise AuditError("payload extent/cap mismatch")
    ph=sha_bytes(blob)
    if ph!=m["payload"].get("sha256"): raise AuditError("payload hash mismatch")
    byname={}; cursor=0
    for a in arrays:
        name=a.get("name"); dtype=a.get("dtype")
        if not isinstance(name,str) or name in byname: raise AuditError(f"duplicate/bad descriptor {name!r}")
        if dtype not in DTYPE_SIZE or int(a.get("element_bytes",-1))!=DTYPE_SIZE[dtype]: raise AuditError(f"unsupported descriptor dtype/stride: {name}")
        count=int(a.get("count",-1)); off=int(a.get("offset",-1)); size=int(a.get("byte_count",-1))
        if count<0 or off!=cursor or size!=count*DTYPE_SIZE[dtype] or off+size>len(blob): raise AuditError(f"bad payload extent for {name}")
        if sha_bytes(blob[off:off+size])!=a.get("sha256"): raise AuditError(f"array hash mismatch for {name}")
        cursor+=size; byname[name]=a
    if cursor!=len(blob): raise AuditError("unaccounted payload bytes")
    if len(arrays)!=8+5*budget: raise AuditError("expected 8 fixed plus 5 per trial arrays")
    fixed=["armijo_entry_global_status","armijo_trial_status_checkpoints","armijo_trial_global_status_after_decision","armijo_final_global_status","armijo_final_fem_status","armijo_final_metrics","armijo_final_fem_line_search","armijo_final_environment_line_search"]
    if any(n not in byname for n in fixed): raise AuditError("missing fixed descriptor")
    if byname["armijo_trial_status_checkpoints"].get("shape")!=[budget,8,envs] or byname["armijo_trial_global_status_after_decision"].get("shape")!=[budget,envs]:
        raise AuditError("status array shape mismatch")
    if byname["armijo_trial_status_checkpoints"]["count"]!=budget*8*envs or byname["armijo_trial_global_status_after_decision"]["count"]!=budget*envs:
        raise AuditError("status array count mismatch")
    for i in range(budget):
        for s,dtype in [("metrics_loop_entry","float32x4"),("metrics_predecision","float32x4"),("metrics_postdecision","float32x4"),("fem_line_search_postdecision","float32x4"),("environment_line_search_postdecision","float32")]:
            n=f"trial_{i}_{s}"
            if n not in byname or byname[n]["dtype"]!=dtype: raise AuditError(f"missing/mistyped array {n}")
    final_fem=byname["armijo_final_fem_line_search"]
    if final_fem["dtype"]!="float32x4" or final_fem["count"]%envs: raise AuditError("bad final FEM line-search layout")
    objects=final_fem["count"]//envs
    if objects<1: raise AuditError("invalid inferred object count")
    alpha_bytes=max(envs*4,16) # checkedBytes minimum allocation
    fixed_sum=sum(byname[n]["byte_count"] for n in ["armijo_entry_global_status","armijo_final_global_status","armijo_final_fem_status","armijo_final_metrics","armijo_final_fem_line_search","armijo_final_environment_line_search"])
    trial_sum=(byname["armijo_trial_status_checkpoints"]["byte_count"] + byname["armijo_trial_global_status_after_decision"]["byte_count"] + sum(byname[f"trial_{i}_{s}"]["byte_count"] for i in range(budget) for s in ["metrics_loop_entry","metrics_predecision","metrics_postdecision","fem_line_search_postdecision","environment_line_search_postdecision"]))
    fixed_formula=160*envs+16*envs*objects+alpha_bytes
    trial_formula=480*envs+16*envs*objects+alpha_bytes
    if fixed_sum!=fixed_formula or trial_sum!=budget*trial_formula or len(blob)!=fixed_sum+trial_sum:
        raise AuditError(f"descriptor sum/formula mismatch fixed={fixed_sum}/{fixed_formula}, trial={trial_sum}/{budget*trial_formula}")
    if byname["armijo_final_environment_line_search"]["count"]!=max(envs,4): raise AuditError("wrong final alpha padded extent")
    for i in range(budget):
        if byname[f"trial_{i}_environment_line_search_postdecision"]["count"]!=max(envs,4): raise AuditError(f"wrong alpha padded extent trial {i}")
        if byname[f"trial_{i}_fem_line_search_postdecision"]["count"]!=envs*objects: raise AuditError(f"wrong object-sized FEM alpha extent trial {i}")
    def arrbytes(name): 
        a=byname[name]; return blob[a["offset"]:a["offset"]+a["byte_count"]]
    def f4(name,idx=0): return struct.unpack_from("<4f",arrbytes(name),idx*16)
    def status_at(name,idx=0): return decode_status(arrbytes(name)[idx*48:(idx+1)*48])
    trials=[]
    for i in range(budget):
        loop=f4(f"trial_{i}_metrics_loop_entry"); pre=f4(f"trial_{i}_metrics_predecision"); post=f4(f"trial_{i}_metrics_postdecision")
        stage_rows=[]
        for j,stage in enumerate(STAGES):
            st=status_at("armijo_trial_status_checkpoints",i*8+j)
            stage_rows.append({"stage":stage,**st})
        first=next((s for s in stage_rows if s["code"]!=0),None)
        g=status_at("armijo_trial_global_status_after_decision",i)
        row={"trial_index":i,"entry_alpha":jnum(loop[2]),"tested_alpha":jnum(pre[2]),
          "base_l2":jnum(pre[0]),"trial_l2":jnum(pre[1]),"acceptance_code":jnum(pre[3]),
          "loop_entry_inactive":loop[3]>.5,"first_nonzero_stage":first["stage"] if first else None,
          "first_status":first,"all_stage_statuses_success":first is None,
          "global_status_after_decision":g,"postdecision_metrics":[jnum(float(x)) for x in post],
          "postdecision_fem_line_search":[jnum(float(x)) for x in f4(f"trial_{i}_fem_line_search_postdecision")],
          "postdecision_environment_line_search_raw":[jnum(struct.unpack_from("<f",arrbytes(f"trial_{i}_environment_line_search_postdecision"),4*k)[0]) for k in range(max(envs,4))]}
        if first and first["stage"]=="deformable_self_narrowphase" and first["code"]==6:
            di=first["diagnostics"]
            if all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in di):
                row["candidate_failure_decode"]={"candidate_identity_xy":[int(di[0]),int(di[1]),],
                  "point_vertex_index":int(di[2]),"vertex_triangle_code":int(di[3]),
                  "direction":int(di[3])-3,
                  "interpretation":"Matches the vertexTriangleCCD uncertified branch in frozen source120: direction 0 checks left vertices against right triangle; direction 1 checks right vertices against left triangle. This is lack of CCD certification, not proof of hit or penetration."}
        trials.append(row)
    entry=status_at("armijo_entry_global_status")
    final_global=status_at("armijo_final_global_status"); final_fem_status=status_at("armijo_final_fem_status")
    final_metrics=f4("armijo_final_metrics")
    alpha_count=byname["armijo_final_environment_line_search"]["count"]
    final_alpha=[jnum(struct.unpack_from("<f",arrbytes("armijo_final_environment_line_search"),4*k)[0]) for k in range(alpha_count)]
    return {"capture":{k:m.get(k) for k in ["source_physics_fingerprint","device_program_fingerprint","root","iteration","line_search_budget","environment_count","node_count","tetrahedron_count","particle_count","contact_pair_count","deformable_contact_capacity","cohesive_face_count","unknown_count"]},
      "payload":{"bytes":len(blob),"sha256":ph,"array_count":len(arrays),"inferred_object_count":objects,"fixed_bytes":fixed_sum,"per_trial_bytes":trial_sum//budget,"formula_fixed_bytes":fixed_formula,"formula_per_trial_bytes":trial_formula},
      "entry_status":entry,"final_global_status":final_global,"final_fem_status":final_fem_status,
      "final_metrics":[jnum(float(x)) for x in final_metrics],"final_environment_line_search":final_alpha,"trials":trials,
      "source_branch_evidence":{"file":"matter/src/metal/contact.metalinc","lines":[3962,3978],"sha256":None,
       "meaning":"vertexTriangleCCD returns certified; when false, production records CONTACT_FAILURE with (candidate.identity.x, candidate.identity.y, vertexIndex, 3+direction) and exits before measuring a residual."}}

def armijo_rows(run:Path):
    rows=[]
    for line in (run/"stderr.log").read_text().splitlines():
        prefix="fem_residual_armijo_iteration="
        if line.startswith(prefix): rows.append(strict_json_text(line[len(prefix):],f"{run.name} Armijo row"))
    return rows

def filtered_log(path:Path)->bytes:
    lines=path.read_text().splitlines()
    keep=[x for x in lines if not x.startswith("fem_coupled_trial_summary_")]
    return ("\n".join(keep)+("\n" if keep else "")).encode()

def validate_run(run:Path):
    p=strict_json_file(run/"prediction.json"); r=strict_json_file(run/"receipt.json")
    if (r.get("schema")!="numi.cardboard.native-run-receipt.v2" or r.get("execution_started") is not True or r.get("source_inputs_unchanged") is not True or r.get("changed_inputs")!=[] or r.get("timed_out") is not False or r.get("termination_errors")!=[] or r.get("exit_code")!=1):
        raise AuditError(f"{run.name}: receipt not an unchanged, completed expected-failure run")
    snap=Path(p["source_snapshot"]); runtime=Path(p["runtime_dir"]); buildfile=Path(p["build_provenance"]); build=strict_json_file(buildfile)
    if build.get("success") is not True or build.get("source_snapshot")!=str(snap): raise AuditError(f"{run.name}: build provenance not successful/bound")
    for rel,expected in build.get("source_sha256",{}).items():
        f=snap/rel
        if not f.is_file() or sha_file(f)!=expected: raise AuditError(f"{run.name}: snapshot source mismatch {rel}")
    for text,expected in build.get("artifacts_sha256",{}).items():
        f=Path(text)
        if not f.is_file() or sha_file(f)!=expected: raise AuditError(f"{run.name}: artifact mismatch {f}")
    for key,expected in p.get("inputs_sha256",{}).items():
        if key.startswith("source/"): f=snap/key[7:]
        elif key.startswith("runtime/"): f=runtime/key[8:]
        elif key=="external/source-build.json": f=buildfile
        else: raise AuditError(f"{run.name}: unknown input key {key}")
        if not f.is_file() or sha_file(f)!=expected: raise AuditError(f"{run.name}: input digest mismatch {key}")
    rows=armijo_rows(run)
    if len(rows)!=32: raise AuditError(f"{run.name}: expected 32 Armijo records, got {len(rows)}")
    return {"run":run.name,"prediction":p,"receipt":r,"source_build_sha256":sha_file(buildfile),
      "source_file_count_verified":len(build.get("source_sha256",{})),"artifact_count_verified":len(build.get("artifacts_sha256",{})),"armijo_rows":rows}

def physical_argv(p):
    out=[]; skip=False
    for i,x in enumerate(p["argv"]):
        if skip: out.append("<diagnostic-state-dir>");skip=False;continue
        if x=="--diagnostic-state-output": out.append(x);skip=True
        elif i==0: out.append(Path(x).name)
        elif x.endswith(".nmatter"): out.append("material-sha256:"+sha_file(Path(x)))
        else: out.append(x)
    return out

def physics_env(p):
    return {k:v for k,v in (p.get("environment") or {}).items() if not k.startswith("NM_FEM_COUPLED_TRIAL_SUMMARY_")}

def fingerprint_only(a:bytes,b:bytes)->bool:
    return len(a)==len(b) and len(a)>=20 and a[:12]==b[:12] and a[20:]==b[20:]

def diagnostic_tree(a:Path,b:Path):
    da=a/"diagnostic-state"/"dense"; db=b/"diagnostic-state"/"dense"
    ma={str(p.relative_to(da)):p for p in da.rglob("*") if p.is_file()}
    mb={str(p.relative_to(db)):p for p in db.rglob("*") if p.is_file()}
    if set(ma)!=set(mb): return {"pass":False,"reason":"inventory mismatch","left_only":sorted(set(ma)-set(mb)),"right_only":sorted(set(mb)-set(ma))}
    diffs=[]; fp=[]
    for rel in sorted(ma):
        x=ma[rel].read_bytes();y=mb[rel].read_bytes()
        if rel.endswith(".log"): x=filtered_log(ma[rel]);y=filtered_log(mb[rel])
        if x==y: continue
        if rel.endswith(".bin") and fingerprint_only(x,y): fp.append(rel);continue
        diffs.append({"path":rel,"left_sha256":sha_bytes(x),"right_sha256":sha_bytes(y)})
    return {"pass":not diffs,"file_count":len(ma),"differences":diffs,"device_fingerprint_only_binary_differences":fp}

def compare(a_run:Path,b_run:Path,a:dict,b:dict):
    stdout=(a_run/"stdout.log").read_bytes()==(b_run/"stdout.log").read_bytes()
    stderr=filtered_log(a_run/"stderr.log")==filtered_log(b_run/"stderr.log")
    tree=diagnostic_tree(a_run,b_run)
    av=physical_argv(a["prediction"]);bv=physical_argv(b["prediction"])
    ae=physics_env(a["prediction"]);be=physics_env(b["prediction"])
    rows=a["armijo_rows"]==b["armijo_rows"]
    return {"runs":[a_run.name,b_run.name],"physical_argv_equal":av==bv,"normalized_argv":av,
      "non_summary_environment_equal":ae==be,"armijo_row_count":len(a["armijo_rows"]),
      "armijo_rows_exact_equal":rows,"stdout_exact_equal":stdout,
      "stderr_exact_equal_after_filtering_only_summary_events":stderr,"diagnostic_state":tree,
      "production_payload_parity":av==bv and ae==be and rows and stdout and stderr and tree["pass"],
      "physical_verdict":{"exit_codes":[a["receipt"]["exit_code"],b["receipt"]["exit_code"]],
       "both_failed_fixture":True,"successful_simulation_claim":False}}

def events(run:Path):
    out=[]
    for line in (run/"stderr.log").read_text().splitlines():
        if line.startswith("fem_coupled_trial_summary_"):
            k,v=line.split("=",1);out.append({"event":k,"value":strict_json_text(v,k)})
    return out

def run_audit():
    a=validate_run(RUN395); b=validate_run(RUN393); c=validate_run(RUN391)
    raw_path=RUN395/"trial-summary.summary.json"; raw=raw_path.read_text()
    fixed,repair=repair_manifest(raw)
    adjudication=Path(__file__).parent/"395-manifest-adjudicated.json"
    adjudication.write_text(fixed+"\n")
    summary=verify_payload(RUN395,fixed)
    source_path=Path(a["prediction"]["source_snapshot"])/"matter/src/metal/contact.metalinc"
    summary["source_branch_evidence"]["sha256"]=sha_file(source_path)
    ev395=events(RUN395);ev393=events(RUN393)
    if not any(x["event"]=="fem_coupled_trial_summary_complete" for x in ev395): raise AuditError("395 completion event missing")
    if not any(x["event"]=="fem_coupled_trial_summary_skipped" for x in ev393): raise AuditError("393 skip event missing")
    if (RUN393/"trial-summary.summary.json").exists() or (RUN393/"trial-summary.summary.bin").exists(): raise AuditError("393 unexpectedly has payload")
    comps={"395_vs_391":compare(RUN395,RUN391,a,c),"395_vs_393":compare(RUN395,RUN393,a,b),"393_vs_391":compare(RUN393,RUN391,b,c)}
    if not all(v["production_payload_parity"] for v in comps.values()): raise AuditError("physical output parity mismatch")
    failed=[r for r in summary["trials"] if not r["all_stage_statuses_success"]]
    feasible=[r for r in summary["trials"] if r["all_stage_statuses_success"]]
    if len(failed)!=22 or [r["trial_index"] for r in feasible]!=[22,23]: raise AuditError("unexpected feasibility pattern")
    r22=summary["trials"][22];r23=summary["trials"][23]
    base=struct.unpack("<f",struct.pack("<f",float(r22["base_l2"])))[0]
    trial=struct.unpack("<f",struct.pack("<f",float(r22["trial_l2"])))[0]
    bb=struct.unpack("<I",struct.pack("<f",base))[0];tb=struct.unpack("<I",struct.pack("<f",trial))[0]
    summary["feasible_trial_descent"]={"trial_indices":[22,23],"both_status_clear_through_measurement":True,
      "base_l2":base,"base_float32_bits":f"0x{bb:08x}","trial22_l2":trial,
      "trial23_l2":float(r23["trial_l2"]),"trial_float32_bits":f"0x{tb:08x}",
      "trial_minus_base":trial-base,"relative_increase":(trial-base)/base,"ulp_steps":tb-bb,
      "tested_alphas":[r22["tested_alpha"],r23["tested_alpha"]],
      "acceptance_codes":[r22["acceptance_code"],r23["acceptance_code"]],
      "interpretation":"The first status-clear samples are rejected: the trial norm is one FP32 ULP above base and unchanged when alpha is halved. This is no-descent evidence, not a measured residual noise floor."}
    return {"schema":"numi.cardboard.all_trial_summary_adjudication.v1",
      "status":"PASS_DIAGNOSTIC_PAYLOAD_AND_PARITY; EXPECTED_PHYSICAL_FAILURE_PRESERVED",
      "raw_capture_manifest":{"path":str(raw_path),**repair,"adjudication_copy":str(adjudication),
        "adjudication_copy_sha256":sha_file(adjudication),
        "qualification":"Raw native JSON remains invalid and untouched. Table decoded only from one-byte-repaired adjudication copy; binary independently hash/extent checked."},
      "capture":summary,"events":{"395":ev395,"393":ev393},
      "lineage":{k:{"source":v["prediction"]["source_snapshot"],"source_build_sha256":v["source_build_sha256"],
                    "source_files_verified":v["source_file_count_verified"],"artifacts_verified":v["artifact_count_verified"]}
                 for k,v in [("395",a),("393",b),("391",c)]},
      "parity":comps,
      "physical_verdict":{"exit_codes":{"395":a["receipt"]["exit_code"],"393":b["receipt"]["exit_code"],"391":c["receipt"]["exit_code"],
        "all_no_timeout":True,"fixture_failed_at_newton9":True,"simulation_qualified":False}}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path(__file__).parent/"report.json");args=ap.parse_args()
    report=run_audit();args.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(f"PASS payload/parity audit; physical failure preserved; report={args.out}")
if __name__=="__main__":
    try: sys.exit(main())
    except AuditError as e: print(f"FAIL: {e}",file=sys.stderr);sys.exit(2)

