#!/usr/bin/env python3
"""Static source-bound checks for coupled trial capture overlay 002."""
from pathlib import Path
import hashlib, json, sys

ROOT=Path("/Users/home/cardboard-implementation-20261014")
BASE=ROOT/"snapshots/source-private-trial-and-material-sweep-092/matter/src/runtime.mm"
OVERLAY=ROOT/"analysis/coupled-trial-capture-overlay-002/overlay/matter/src/runtime.mm"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    if not c: raise AssertionError(m)

def main():
    base=BASE.read_text(); text=OVERLAY.read_text()
    req(sha(BASE)=="4b8dc05747e3224e8bc82692c795a56f3b3cfd7ad5ea067fd151c056fdcb31e4","wrong frozen-092 base")
    req("trialIndex != femCoupledTrialCaptureTrial" in text,"residual stages are not selected-trial gated")
    req(text.count("captureTrialResidualStage(\n") == 5,"expected five residual-stage callsites")
    req(text.count("\", trialIndex)") == 5,"each stage must pass trialIndex")
    req("const std::uint32_t trialIndex) -> bool" in text,"measurement gate signature missing")
    req("captureTrialMeasurement(trialIndex)" in text,"measurement trial argument missing")
    req('push("rigid_states_trial"' in text and 'push("rigid_candidate_trial"' in text,"trial rigid arrays missing")
    req(text.count('push("rigid_states_trial"')==1 and text.count('push("rigid_candidate_trial"')==1,"duplicate rigid trial capture")
    rigid=text.index('push("rigid_states_trial"')
    post=text.index("if (postRebuild) {",text.index('push("fem_armijo_candidate_nodes"'))
    req(rigid < post,"pre context still omits private rigid candidate")
    req("prePhaseMaximum = preContextMaximum" in text and "preContextMaximum = 71u" in text,"pre array bound missing")
    req("postPhaseMaximum =" in text and "bothPhaseMaximum =" in text,"phase array bounds missing")
    req("bothPhaseMaximum == 136u" in text and "postPhaseMaximum == 109u" in text,"derived exact both/post count mismatch")
    req("bothPhaseMaximum <= captureArrayLimit" in text,"selected-path bound not checked under array cap")
    req("32u * 1024u * 1024u" in text and "aggregate_readback_cap_bytes" in text,"32 MiB payload cap changed")
    req("femCoupledTrialCaptureArrays.size() + present >" in text and "captureArrayLimit = 192u" in text,"array cap changed")
    req("selected_phase_array_count_exceeds_derived_bound" in text,"derived bound not fail-closed")
    req(base != text,"overlay equals base")
    result={"schema":"numi.matter.coupled_trial_capture_overlay_002_static_check.v1","base_sha256":sha(BASE),"overlay_sha256":sha(OVERLAY),"derived_max_arrays":{"pre":71,"post":109,"both":136},"payload_cap_bytes":33554432,"array_cap":192,"checks":17,"result":"PASS_STATIC_ONLY","native_or_physical_qualification":False}
    print(json.dumps(result,sort_keys=True,separators=(",",":")))
if __name__=="__main__":
    try: main()
    except Exception as e: print("FAIL:",e,file=sys.stderr);sys.exit(1)
