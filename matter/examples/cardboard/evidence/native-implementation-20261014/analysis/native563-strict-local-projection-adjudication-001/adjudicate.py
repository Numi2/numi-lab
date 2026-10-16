from __future__ import annotations
import hashlib, json, math, struct
from pathlib import Path
import numpy as np

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
OUT = Q / "analysis/native563-strict-local-projection-adjudication-001"
OUT.mkdir(parents=True, exist_ok=True)
INPUT = Q / "analysis/native547-critical23-sweep-input-002/input.json"
REF = Q / "analysis/native547-critical23-sweep-input-002/stable-fp64-reference.jsonl"
PAIR = Q / "analysis/native547-critical23-sweep-input-002/pair-selection.json"
INPUT_BUILDER = Q / "analysis/native547-critical23-sweep-input-002/build_input.py"
STRICT_HELPER = Q / "snapshots/source-cohesive-quadrature-area-205/matter/tools/cardboard_compensated_fp64_reference.py"
STABLE_REF = Q / "snapshots/source-stable-green-reference-206-buildfix-002/matter/tools/cardboard_bilinear_fp64_reference.py"
FINE_REPORT = Q / "analysis/native531-stable-green-strain-reference-017-001/extended23-v2/report.json"
OLD = Q / "native/critical23-power-sweep-547/output/paper-points-exact_package_budget32.jsonl"
RUN = Q / "native/strict-local-projection-critical23-fixed-563"
CMD = RUN / "command.json"
RECEIPT = RUN / "receipt.json"
ACCURATE = RUN / "output/paper-points-accurate32.jsonl"
STRICT = RUN / "output/paper-points-strict_state_2e-10_budget32.jsonl"
SUMMARY = RUN / "output/source547-strict-state-certification.json"
SOURCE = Q / "snapshots/source-local-projection-state-certification-220-endpoint-identity-repair-003/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm"
BUILD = Q / "build-source220-endpoint-identity-repair-003/numi-matter-cardboard-bilinear15-capture-diagnostic-check"
METALLIB = Q / "build-source220-endpoint-identity-repair-003/shaders/cardboard_bilinear15_capture_diagnostic.metallib"
EXPECTED = {
    "input_sha256": "a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e",
    "reference_sha256": "2893d2b988feae51fe6ecbe3a7a71ecb7a19d1f9b011a3957b52df54c4bd96e0",
    "pair_sha256": "36dfab1b02c443ef9ee77df2cada5e7eafa9abe3450c1d33208016357ceeb452",
    "input_builder_sha256": "7654967ad7b8f9e145e97b228075e8372b1e72fede8597e91c5b5c1a63516f3a",
    "strict_helper_sha256": "760ea825449857d565c78e754c755845e4547134b381fa167d02aab2ae63720c",
    "stable_reference_source_sha256": "8064ba89f0c4b10413937cbf2a5886a9f7de0e4b02358988294620ebae51788f",
    "fine_report_sha256": "280b7ce27cbea90ea6a033cf9d3ca93585654ab3174b168bbd81e907e092ce63",
    "source_sha256": "dd017475bcbc7df047d3a8ac66d9729c3ac0d85552532c23ba85e908c5750971",
    "binary_sha256": "412b901a568c5a5de176fb01668a9d5575be938af243e921cbed79b43e752e12",
    "metallib_sha256": "ce7b2e8af1a9fe1c528ffda664c17758f3870772f67600dfbc4890f076f8a61b",
    "source_frozen_sha256": "ce4db34124a09c8d1356af11c22a42cad9779629cf1800965ed6e5878d5f4639",
    "package_manifest_sha256": "9433667e5bc44cda47e5dd07140525c14c2e27a431896fea6e8d596423ee9e51",
}
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def f32_word(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]
def as_words(xs):
    return [f32_word(x) for x in xs]
def nrm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))
def relerr(actual, expected):
    a=np.asarray(actual,dtype=np.float64).reshape(-1)
    b=np.asarray(expected,dtype=np.float64).reshape(-1)
    return nrm(a-b)/nrm(b)

for p, key in ((INPUT,"input_sha256"),(REF,"reference_sha256"),(PAIR,"pair_sha256"),
               (INPUT_BUILDER,"input_builder_sha256"),(STRICT_HELPER,"strict_helper_sha256"),
               (STABLE_REF,"stable_reference_source_sha256"),(FINE_REPORT,"fine_report_sha256"),
               (SOURCE,"source_sha256"),(BUILD,"binary_sha256"),(METALLIB,"metallib_sha256")):
    actual=sha(p)
    if actual != EXPECTED[key]:
        raise SystemExit(f"{key} mismatch: {actual}")
cmd=json.loads(CMD.read_text())
receipt=json.loads(RECEIPT.read_text())
summary=json.loads(SUMMARY.read_text())
if cmd["source_frozen_sha256"] != EXPECTED["source_frozen_sha256"]:
    raise SystemExit("source frozen map mismatch")
if cmd["package_manifest_sha256"] != EXPECTED["package_manifest_sha256"]:
    raise SystemExit("package manifest mismatch")
if cmd["input_sha256"] != EXPECTED["input_sha256"]:
    raise SystemExit("native command input mismatch")
if receipt["exit_code"] != 0 or receipt["timed_out"] or not receipt["all_inputs_unchanged"]:
    raise SystemExit("native receipt is not a clean unchanged-input completion")
if receipt["physical_transactions"] != 0 or receipt["full_solver_qualification"]:
    raise SystemExit("native run has unexpected physical/full-solver status")

input_doc=json.loads(INPUT.read_text())
refs=read_jsonl(REF)
old_records=[x for x in read_jsonl(OLD) if x.get("record")=="point"]
if len(input_doc["rows"]) != 1472 or len(refs) != 1472 or len(old_records) != 1472:
    raise SystemExit("full source547 input/reference/native row counts differ")
indices=[256,257,282,283,284,285,286,287,288,289]
expected=[
    (256,"center",None,0),(257,"center_repeat",None,0),
    (282,"endpoint",22,1),(283,"endpoint",22,-1),
    (284,"endpoint",23,1),(285,"endpoint",23,-1),
    (286,"endpoint",24,1),(287,"endpoint",24,-1),
    (288,"endpoint",25,1),(289,"endpoint",25,-1),
]
if summary["rows"] != 10 or summary["input_sha256"] != EXPECTED["input_sha256"]:
    raise SystemExit("source547 native summary does not bind ten corrected rows")

def point_rows(p):
    parsed=read_jsonl(p)
    hdr=next(x for x in parsed if x.get("record")=="header")
    points=[x for x in parsed if x.get("record")=="point"]
    if hdr["point_input_sha256"] != EXPECTED["input_sha256"] or hdr["record_count"] != 10:
        raise SystemExit(f"output header identity mismatch: {p}")
    if len(points) != 10:
        raise SystemExit(f"output point count mismatch: {p}")
    return hdr,points

headers={}
rows={}
for policy,path in (("accurate32",ACCURATE),("strict_state_2e-10_budget32",STRICT)):
    headers[policy],rows[policy]=point_rows(path)

selected=[]
for position,(idx,label,power,sign) in enumerate(expected):
    src=input_doc["rows"][idx]
    fp=refs[idx]
    if (src["environment"],src["tet_id"],src["material_index"]) != (0,222,0):
        raise SystemExit(f"source row identity mismatch at {idx}")
    if (fp["env"],fp["tet"],fp["material"]) != (0,222,0):
        raise SystemExit(f"reference row identity mismatch at {idx}")
    if (src["label"],src["basis_column"],src["epsilon_power"],src["sign"]) != (label,2,power,sign):
        raise SystemExit(f"input label/endpoint identity mismatch at {idx}")
    if (fp["label"],fp["basis_column"],fp["power"],fp["sign"]) != (label,2,power,sign):
        raise SystemExit(f"FP64 reference endpoint identity mismatch at {idx}")
    if as_words(src["packed_input_f32"]) != [int(w,16) for w in fp["input_f32_hex"]]:
        raise SystemExit(f"exact 47-word packed input mismatch at {idx}")
    if as_words(src["accepted_state_f32"]) != [int(w,16) for w in fp["accepted_f32_hex"]]:
        raise SystemExit(f"exact accepted history mismatch at {idx}")
    if not fp["strict_valid"]:
        raise SystemExit(f"independent FP64 reference root invalid at {idx}")
    # Independently verify endpoint F-high/F-low/Fdot encoding against nominal epsilon.
    if label=="endpoint":
        c=input_doc["rows"][256]["packed_input_f32"]
        x=src["packed_input_f32"]
        eps=math.ldexp(1.0,-power); b=2
        dt=float(c[45])
        high=np.float32(float(c[b])+float(c[9+b])+sign*eps)
        low=np.float32((float(c[b])-float(high))+float(c[9+b])+sign*eps)
        rate=np.float32(float(c[18+b])+sign*eps/dt)
        for got,want,name in ((x[b],high,"Fhigh"),(x[9+b],low,"Flow"),(x[18+b],rate,"Fdot")):
            if f32_word(got)!=f32_word(want):
                raise SystemExit(f"endpoint {name} mismatch at source row {idx}")
        for k,(got,center) in enumerate(zip(x,c)):
            if k not in (b,9+b,18+b) and f32_word(got)!=f32_word(center):
                raise SystemExit(f"undeclared endpoint component changed at row {idx}, slot {k}")
    per_policy={}
    for policy in rows:
        native=rows[policy][position]
        if (native["environment"],native["tet_id"],native["material_index"]) != (0,222,0):
            raise SystemExit(f"native output identity mismatch {policy}/{position}")
        if not native["stress_valid"] or not native["tangent_valid"] or not native["tangent_projection_converged"] or native["projection_failure"] != 0:
            raise SystemExit(f"native local evaluator rejected {policy}/{position}")
        tangent_rel=relerr(native["tangent_pa_per_s"],fp["tangent"])
        stress_rel=relerr(native["direct_stress_pa"],fp["stress"])
        state_hi=np.asarray(native["projected_state_high"],dtype=np.float64)
        state_lo=np.asarray(native["projected_state_low"],dtype=np.float64)
        state_error=nrm(state_hi+state_lo-np.asarray(fp["state"],dtype=np.float64))
        state_rel=state_error/nrm(fp["state"])
        direct_tangent_stress_diff=nrm(np.asarray(native["direct_stress_pa"])-np.asarray(native["tangent_path_stress_pa"]))
        per_policy[policy]={
            "native_projection_residual":float(native["projection_residual"]),
            "native_projection_iterations":int(native["projection_iterations"]),
            "stress_valid":bool(native["stress_valid"]),
            "tangent_valid":bool(native["tangent_valid"]),
            "native_tangent_relative_l2_to_independent_FP64":tangent_rel,
            "native_stress_relative_l2_to_independent_FP64":stress_rel,
            "projected_state_high_plus_low_absolute_l2_to_FP64_root":state_error,
            "projected_state_relative_l2_to_FP64_root":state_rel,
            "direct_vs_tangent_path_stress_absolute_l2":direct_tangent_stress_diff,
            "tangent_gate_1e-3_pass":tangent_rel <= 1e-3,
            "matches_source547_native_baseline_65_words":native["output_f32_bits"]==old_records[idx]["output_f32_bits"],
        }
    selected.append({
        "source_row_index":idx,"label":label,"basis_column":2,"epsilon_power":power,"sign":sign,
        "nominal_epsilon":None if power is None else math.ldexp(1.0,-power),
        "realized_delta_F_basis":None if power is None else float(fp["realized_delta_F"][2][0]),
        "realized_delta_Fdot_basis":None if power is None else float(fp["realized_delta_Fdot"][2]),
        "packed_input_f32_words_exact":True,"accepted_history_f32_words_exact":True,
        "independent_FP64_root_valid":True,"FP64_root_residual":float(fp["scaled_residual"]),
        "FP64_root_dynamic_correction":float(fp["dynamic_correction"]),
        "FP64_root_iterations":int(fp["iterations"]),
        "reference_tangent_norm":nrm(fp["tangent"]),
        "policy_results":per_policy,
    })

aggregate={}
for policy in rows:
    r=selected
    errs=[x["policy_results"][policy]["native_tangent_relative_l2_to_independent_FP64"] for x in r]
    stress=[x["policy_results"][policy]["native_stress_relative_l2_to_independent_FP64"] for x in r]
    states=[x["policy_results"][policy]["projected_state_relative_l2_to_FP64_root"] for x in r]
    proj=[x["policy_results"][policy]["native_projection_residual"] for x in r]
    its=[x["policy_results"][policy]["native_projection_iterations"] for x in r]
    gates=sum(x<=1e-3 for x in errs)
    worst=max(range(len(r)),key=lambda i:errs[i])
    aggregate[policy]={
        "valid_points":len(r),"native_tangent_gate_1e-3_pass_count":gates,
        "max_native_tangent_relative_l2":max(errs),
        "max_native_tangent_error_row":{
            "source_row_index":r[worst]["source_row_index"],
            "label":r[worst]["label"],
            "basis_column":r[worst]["basis_column"],
            "epsilon_power":r[worst]["epsilon_power"],
            "sign":r[worst]["sign"],
        },
        "max_native_stress_relative_l2":max(stress),
        "max_projected_state_relative_l2":max(states),
        "max_native_projection_residual":max(proj),
        "max_projection_iterations":max(its),
        "repeat_center_65word_bitwise_equal":rows[policy][0]["output_f32_bits"]==rows[policy][1]["output_f32_bits"],
        "all_points_bitwise_equal_to_older_native547_accurate_batch":all(x["policy_results"][policy]["matches_source547_native_baseline_65_words"] for x in r),
    }

report={
 "schema":"numi.cardboard.native563-strict-local-projection-adjudication.v1",
 "verdict":"SELECTED_STRICT_STATE_CERTIFICATION_ROWS_PASS_NATIVE_TANGENT_1E-3_GATE",
 "scope":"10 selected env0/tet222/liner rows only: center, exact center repeat, and signed basis-column-2 endpoints at powers22-25",
 "bindings":{
  "source547_input_sha256":sha(INPUT),"source547_reference_sha256":sha(REF),
  "source547_pair_selection_sha256":sha(PAIR),"native547_baseline_jsonl_sha256":sha(OLD),
  "source547_input_builder_sha256":sha(INPUT_BUILDER),
  "FP64_compensated_projection_helper_sha256":sha(STRICT_HELPER),
  "FP64_stable_Green_reference_source_sha256":sha(STABLE_REF),
  "source547_fine_reference_report_sha256":sha(FINE_REPORT),
  "native563_command_sha256":sha(CMD),"native563_receipt_sha256":sha(RECEIPT),
  "native563_accurate_output_sha256":sha(ACCURATE),"native563_strict_output_sha256":sha(STRICT),
  "native563_summary_sha256":sha(SUMMARY),"source220_diagnostic_cpp_sha256":sha(SOURCE),
  "diagnostic_executable_sha256":sha(BUILD),"diagnostic_metallib_sha256":sha(METALLIB),
  "source_frozen_sha256":cmd["source_frozen_sha256"],"package_manifest_sha256":cmd["package_manifest_sha256"],
  "native563_receipt_stdout_sha256":receipt["stdout_sha256"],
 },
 "run":{
  "exit_code":receipt["exit_code"],"timed_out":receipt["timed_out"],
  "all_inputs_unchanged":receipt["all_inputs_unchanged"],
  "physical_transactions":receipt["physical_transactions"],
  "full_solver_qualification":receipt["full_solver_qualification"],
  "source_capture_sha256":input_doc["source_capture_sha256"],
  "input_rows_total":len(input_doc["rows"]),
  "selected_rows":indices,
  "selected_endpoint_contract":"For each endpoint, only F-high slot2, F-low slot11, and corresponding Fdot slot20 differ from the center, and the script independently recomputes their float32 words from epsilon/sign/dt.",
  "strict_policy":{
   "iterations_liner":summary["strict_local_iterations"],
   "iterations_medium":summary["strict_local_iterations"],
   "scaled_correction_tolerance":summary["strict_scaled_correction_tolerance"],
   "projection_residual_tolerance_unchanged":summary["projection_residual_tolerance_unchanged"],
   "center_baseline_iterations":summary["center_trace_iterations_baseline"],
   "center_strict_iterations":summary["center_trace_iterations_strict"],
   "p23plus_baseline_iterations":summary["p23plus_trace_iterations_baseline"],
   "p23plus_strict_iterations":summary["p23plus_trace_iterations_strict"],
   "outer_residual_and_correction_gates_changed":summary["outer_residual_and_correction_gates_changed"],
  },
 },
 "aggregate":aggregate,
 "rows":selected,
 "interpretation":{
  "main":"On these exact ten inputs, the default accurate32 result reproduces the older native547 outputs bit for bit. Its positive p23 endpoint (source row284) misses the native tangent-vs-FP64 1e-3 gate at 1.18003497e-3; the p24 positive endpoint is close at 9.9110093e-4. Enabling the separate strict local state-correction certificate, without changing the residual gate, local iteration budget, equations, or outer acceptance criteria, reduces row284 to 8.76067061e-8 and all ten selected tangent comparisons pass 1e-3. The strict ten-row maximum is 3.09479905e-5 at p25+ row288; row284 falls to 8.76067061e-8 and row286 to 5.29998101e-6.",
  "limits":[
   "This is a selected local material evaluator comparison, not a full coupon/solver qualification or calibration.",
   "It does not establish native finite-difference tangent accuracy or an absolute FP32 noise floor; exact native stress/tangent output was compared with the independent FP64 analytic reference.",
   "The native residual and correction certifications are local projection criteria; no global solver tolerance or physical criterion changed.",
   "The input-selection contract is the corrected source547 v2 input. The earlier source547 v1/invalid 541 endpoint batch is not used.",
   "Native548 is unrelated; parent clarified the intended identity source was corrected native547."
  ]
 }
}
if aggregate["strict_state_2e-10_budget32"]["native_tangent_gate_1e-3_pass_count"] != 10:
    raise SystemExit("strict selected native tangent gate did not fully pass")
if aggregate["accurate32"]["native_tangent_gate_1e-3_pass_count"] != 9:
    raise SystemExit("unexpected default accurate32 gate result")
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
