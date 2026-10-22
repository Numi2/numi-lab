#!/usr/bin/env python3
"""Adversarial regression checks for derived hand muscle-law qualification."""
import copy, hashlib, json, pathlib, shutil
import qualify_hand_muscle_owner_law as qual

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE/"owner-law-derived-checks-001"
SOURCE=HERE/"results"/"scoped-telemetry004-bilateral-finger-flexion-reference-initial-800-step100ms"
ORIGINAL=json.loads((SOURCE/"result.json").read_text())
INVOCATION=json.loads((SOURCE/"invocation.json").read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def case(name, mutate_result=None, mutate_invocation=None):
    folder=ROOT/"tests"/name
    folder.mkdir(parents=True,exist_ok=True)
    result=copy.deepcopy(ORIGINAL)
    inv=copy.deepcopy(INVOCATION)
    if mutate_result: mutate_result(result)
    if mutate_invocation: mutate_invocation(inv)
    rp=folder/"result.json"; ip=folder/"invocation.json"; op=folder/"derived.json"
    rp.write_text(json.dumps(result,indent=2,allow_nan=True)+"\n")
    ip.write_text(json.dumps(inv,indent=2,allow_nan=True)+"\n")
    report=qual.qualify(rp,op,HERE)
    return {"case":name,"derived_status":report["derived_status"],
        "failures":[x["code"] for x in report["failures"]],
        "raw_status_preserved":report["raw_runner_status_preserved"],
        "raw_reasons_preserved":report["raw_runner_reasons_preserved"],
        "result_sha256":digest(rp),"invocation_sha256":digest(ip)}

def muscle(d, idx):
    return next(x for x in d["selected_hand_wrist_actuation"]["selected_muscles"]
                if x["source_actuator_index"]==idx)

def zero_positive(d):
    m=muscle(d,242);m["source_force_n"]=0.0;m["tendon_tension_normalized"]=0.0;m["tendon_tension_n"]=0.0

def wrong_large_force(d):
    muscle(d,242)["source_force_n"]=-1000000.0

def force_on_slack(d):
    m=muscle(d,243);m["source_force_n"]=-1.0;m["tendon_tension_normalized"]=0.01;m["tendon_tension_n"]=1.0

def nonfinite_residual(d):
    muscle(d,242)["normalized_equilibrium_residual"]=float("nan")

def bad_activation(d):
    m=muscle(d,242);m["accepted_activation"]=m["baseline_accepted_activation"]

def bad_timestep(inv):
    i=inv["argv"].index("--muscle-step-seconds");inv["argv"][i+1]="0.00025"

checks=[
    case("valid-owner-law-slack-243-306"),
    case("zero-force-on-positive-tension-state",zero_positive),
    case("large-wrong-source-force",wrong_large_force),
    case("nonzero-force-on-owner-slack-state",force_on_slack),
    case("nonfinite-equilibrium-residual",nonfinite_residual),
    case("activation-state-mismatch",bad_activation),
    case("argv-timestep-disagrees-with-invocation",mutate_invocation=bad_timestep),
]
expected={
    "valid-owner-law-slack-243-306":("PASS_CORE_ONLY_OWNER_LAW",None),
    "zero-force-on-positive-tension-state":("FAIL","positive_tension_has_zero_or_wrong_sign_force"),
    "large-wrong-source-force":("FAIL","source_force_not_binary32_owner_product"),
    "nonzero-force-on-owner-slack-state":("FAIL","slack_branch_has_nonzero_force_or_tension"),
    "nonfinite-equilibrium-residual":("FAIL","accepted_residual_missing_or_nonfinite"),
    "activation-state-mismatch":("FAIL","activation_increment_not_realized"),
    "argv-timestep-disagrees-with-invocation":("FAIL","invocation_timestep_not_bound_to_exact_argv"),
}
errors=[]
for item in checks:
    want_status,want_error=expected[item["case"]]
    if item["derived_status"]!=want_status:
        errors.append(item["case"]+": unexpected status "+item["derived_status"])
    if want_error and want_error not in item["failures"]:
        errors.append(item["case"]+": expected failure absent: "+want_error)
report={"schema":"numi.human.hand-owner-law-adversarial-tests.v1",
    "test_status":"PASS" if not errors else "FAIL",
    "source_native_result_sha256":digest(SOURCE/"result.json"),
    "cases":checks,"failures":errors,
    "scope":"qualification assessor only; no native simulation, no raw evidence modified"}
(ROOT/"test-report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"report":str(ROOT/"test-report.json"),"test_status":report["test_status"],
                  "cases":[{"name":x["case"],"status":x["derived_status"],"failures":x["failures"]} for x in checks],
                  "failures":errors},indent=2))
raise SystemExit(0 if not errors else 1)
