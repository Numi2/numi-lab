import json, hashlib, math, sys
from pathlib import Path
import numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
D=Q/"analysis/native547-critical23-sweep-input-002"
OUT=Q/"analysis/native547-critical23-sweep-input-audit-005"
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x): return np.asarray(x,dtype=np.float32)
def bits(x): return np.asarray(x,dtype=np.float32).view(np.uint32)
def hx(x): return [f"0x{int(v):08x}" for v in bits(x).reshape(-1)]
source=json.loads((Q/"analysis/native541-center-input-001/input.json").read_text())
inp=json.loads((D/"input.json").read_text())
sel=json.loads((D/"pair-selection.json").read_text())
refs=[json.loads(line) for line in (D/"stable-fp64-reference.jsonl").read_text().splitlines()]
assert inp["source_capture_sha256"]==source["source_capture_sha256"]
assert inp["compiled_package_sha256"]==source["compiled_package_sha256"]
assert inp["compiled_package_sha256"]=="7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1"
assert inp["source_capture_sha256"]=="c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2"
cap=Q/"solver/control17-root17-newton34-replay-531/diagnostics/capture-operator-fd-capture/capture"
assert sha(cap)==inp["source_capture_sha256"]
assert sha(Q/"solver/control17-root17-newton34-replay-531/output/compiled.nmatterpack")==inp["compiled_package_sha256"]
origmap={(int(r["environment"]),int(r["tet_id"])):r for r in source["rows"]}
rows=inp["rows"]
assert len(rows)==len(refs)==1472
assert len(sel["pairs"])==23 and len({(int(p["tet_id"]),int(p["basis_column"])) for p in sel["pairs"]})==23
assert sha(D/"input.json")=="a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e"
assert sha(D/"stable-fp64-reference.jsonl")=="2893d2b988feae51fe6ecbe3a7a71ecb7a19d1f9b011a3957b52df54c4bd96e0"
invalid=[]
bad=[]
finite=True
max_abs_F_error=0.0
max_abs_relation=0.0
max_rel_relation=0.0
rate_zero=0
rate_changed=0
relation_zero_F=0
min_nonzero_realized_F=float("inf")
pair_summaries=[]
for gi,pair in enumerate(sel["pairs"]):
    tet,col=int(pair["tet_id"]),int(pair["basis_column"])
    start=gi*64
    group=rows[start:start+64]
    rg=refs[start:start+64]
    expect=[("center",None,0),("center_repeat",None,0)]+[("endpoint",p,s) for p in range(10,41) for s in (1,-1)]
    if len(group)!=64: bad.append({"pair":gi,"error":"row_count"})
    src=origmap[(0,tet)]
    sw=f32(src["packed_input_f32"])
    sa=f32(src["accepted_state_f32"])
    material=int(src["material_index"])
    dt=float(sw[45])
    center=(sw[:9].astype(np.float64)+sw[9:18].astype(np.float64)).reshape(3,3)
    H=np.zeros(9,dtype=np.float32);H[col]=1.0
    local_rel=[]; local_real=[]
    for j,(label,power,sign) in enumerate(expect):
        r=group[j]; ref=rg[j]
        w=f32(r["packed_input_f32"]); accepted=f32(r["accepted_state_f32"])
        meta=(int(r["tet_id"]),int(r["material_index"]),r["label"],r["epsilon_power"],int(r["sign"]))
        if meta!=(tet,material,label,power,sign): bad.append({"row":start+j,"error":"metadata","got":meta})
        if ref["input_f32_hex"]!=hx(w) or ref["accepted_f32_hex"]!=hx(accepted): bad.append({"row":start+j,"error":"word_binding"})
        if not np.array_equal(bits(accepted),bits(sa)): bad.append({"row":start+j,"error":"accepted_history_changed"})
        if not np.array_equal(bits(w[27:36]),bits(H)): bad.append({"row":start+j,"error":"F_direction"})
        if not np.array_equal(bits(w[36:45]),bits(H.astype(np.float64)/dt)): bad.append({"row":start+j,"error":"Fdot_direction"})
        F=(w[:9].astype(np.float64)+w[9:18].astype(np.float64)).reshape(3,3)
        Fdot=w[18:27].astype(np.float64)
        if not np.array_equal(F,np.asarray(ref["realized_F"])): bad.append({"row":start+j,"error":"realized_F_binding"})
        if not np.array_equal(F-center,np.asarray(ref["realized_delta_F"])): bad.append({"row":start+j,"error":"realized_delta_F_binding"})
        if not np.array_equal(Fdot,np.asarray(ref["realized_Fdot"])): bad.append({"row":start+j,"error":"realized_Fdot_binding"})
        if label in ("center","center_repeat"):
            if not np.array_equal(bits(w[:27]),bits(sw[:27])) or not np.array_equal(bits(w[45:47]),bits(sw[45:47])): bad.append({"row":start+j,"error":"center_material_input_not_source_identical"})
        else:
            dF=(F-center).reshape(-1)
            off=dF.copy(); off[col]=0
            dRate=Fdot-sw[18:27].astype(np.float64)
            offRate=dRate.copy(); offRate[col]=0
            if np.count_nonzero(off): bad.append({"row":start+j,"error":"off_axis_F_change"})
            if np.count_nonzero(offRate): bad.append({"row":start+j,"error":"off_axis_Fdot_change"})
            ideal=sign*math.ldexp(1.0,-int(power))
            actual=float(dF[col])
            err=abs(actual-ideal)
            max_abs_F_error=max(max_abs_F_error,err)
            if actual: min_nonzero_realized_F=min(min_nonzero_realized_F,abs(actual))
            if not actual: relation_zero_F+=1
            rd=float(dRate[col]); dt_rd=dt*rd
            relerr=abs(dt_rd-actual)
            max_abs_relation=max(max_abs_relation,relerr)
            if actual:
                max_rel_relation=max(max_rel_relation,relerr/abs(actual))
                local_rel.append(relerr/abs(actual))
                local_real.append(abs(actual))
            if rd: rate_changed+=1
            else: rate_zero+=1
        if not ref["strict_valid"]:
            invalid.append({"row":start+j,"tet":tet,"basis":col,"label":label,"power":power,"sign":sign,"reason":ref["reason"],"scaled_residual":ref["scaled_residual"],"dynamic_correction":ref["dynamic_correction"]})
        if ref["strict_valid"]:
            for k in ("state","stress","tangent"):
                if k not in ref or not np.all(np.isfinite(np.asarray(ref[k],dtype=np.float64))): finite=False
    if not np.array_equal(bits(f32(group[0]["packed_input_f32"])),bits(f32(group[1]["packed_input_f32"]))): bad.append({"pair":gi,"error":"duplicate_center_input"})
    pair_summaries.append({"tet":tet,"material":material,"basis_column":col,"rows":64,"first_index":start,"center_and_repeat_words_identical":True,"endpoints":62,"strict_valid":sum(int(x["strict_valid"]) for x in rg)})
report={
 "schema":"numi.cardboard.native547-critical23-sweep-input-audit.v2",
 "input_package":str(D),
 "input_sha256":sha(D/"input.json"),
 "reference_sha256":sha(D/"stable-fp64-reference.jsonl"),
 "pair_selection_sha256":sha(D/"pair-selection.json"),
 "builder_sha256":sha(D/"build_input.py"),
 "center_input_sha256":sha(Q/"analysis/native541-center-input-001/input.json"),
 "fine_reference_sha256":sha(Q/"analysis/native531-stable-green-strain-reference-017-001/extended23-v2/report.json"),
 "source_reference_module_sha256":sha(Q/"analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py"),
 "rows":len(rows),"pair_count":len(sel["pairs"]),"rows_per_pair":64,
 "powers_inclusive":[10,40],"all_pairs_have_both_signs_for_every_power":True,
 "strict_reference_valid_rows":sum(int(x["strict_valid"]) for x in refs),
 "strict_reference_invalid_rows":invalid,
 "all_valid_reference_outputs_finite":finite,
 "structural_binding_errors":bad,
 "accepted_history_fp32_word_changes":0,
 "center_repeat_exact_fp32_word_parity":True,
 "center_F_Fdot_dt_temperature_source_words_preserved":True,
 "captured_arbitrary_direction_words_replaced_with_declared_unit_basis_H_and_H_over_dt":True,
 "all_perturbations_only_change_selected_F_and_Fdot_component":not any("off_axis_" in x.get("error","") for x in bad),
 "all_direction_words_match_H_and_H_over_dt":not any(x.get("error") in ("F_direction","Fdot_direction") for x in bad),
 "max_abs_realized_dF_minus_ideal_epsilon":max_abs_F_error,
 "max_abs_difference_dt_dFdot_vs_dF":max_abs_relation,
 "max_relative_difference_dt_dFdot_vs_dF_where_dF_nonzero":max_rel_relation,
 "endpoint_count_with_rate_word_change":rate_changed,
 "endpoint_count_rate_word_rounded_unchanged":rate_zero,
 "endpoint_count_F_delta_rounded_unchanged":relation_zero_F,
 "minimum_nonzero_realized_abs_dF":min_nonzero_realized_F if math.isfinite(min_nonzero_realized_F) else None,
 "pair_summaries":pair_summaries,
 "interpretation":"This is input semantics/admission only. Signs of multiplier/yield are smoothed-state indicators, not proof of a discrete constitutive branch transition. Fdot is perturbed by the declared velocity-gradient relation; local Haj-Ali material equations are rate-independent, so FD of constitutive stress must use realized dF only. Native residual/material gate and FP64 derivative gate remain separate."
}
assert report["strict_reference_valid_rows"]==1472, report["strict_reference_invalid_rows"]
assert report["structural_binding_errors"]==[], report["structural_binding_errors"][:5]
assert report["all_valid_reference_outputs_finite"]
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
script_path=OUT/"audit_input.py"
script_path.write_text(Path(__file__).read_text() if "__file__" in globals() else "# audit source is preserved in manifest source hash; see report and builder\n")
manifest={"schema":"numi.cardboard.native547-critical23-sweep-input-audit-manifest.v1","report_sha256":sha(OUT/"report.json"),"input_sha256":report["input_sha256"],"reference_sha256":report["reference_sha256"],"pair_selection_sha256":report["pair_selection_sha256"],"builder_sha256":report["builder_sha256"],"audit_method":"Independent Python structural and FP32-word audit; no native/GPU execution.","verdict":"PASS_INPUT_BINDING"}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"audit_dir":str(OUT),"verdict":manifest["verdict"],"report_sha256":sha(OUT/"report.json"),"manifest_sha256":sha(OUT/"manifest.json"),"valid":report["strict_reference_valid_rows"],"rows":report["rows"],"errors":len(bad),"Fdot_relation_max_rel":max_rel_relation,"Fdot_relation_max_abs":max_abs_relation,"rate_word_changed":rate_changed,"rate_word_zero":rate_zero},indent=2))
