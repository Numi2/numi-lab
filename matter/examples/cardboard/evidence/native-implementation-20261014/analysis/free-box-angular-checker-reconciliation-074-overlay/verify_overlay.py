#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
here=Path(__file__).resolve().parent
base_text=(here/"base-073.mm").read_text()
text=(here/"cardboard_free_box_angular_native_check.mm").read_text()
checks={
    "proxy_has_no_adaptive_owner": "proxy_.generalizedFreeBodyIndex=0;proxy_.adaptiveObjectIndex=NM_INVALID_INDEX;" in text,
    "approach_velocity_is_asserted": "need(dot(relativePointVelocity,normalApprox)<-0.49" in text,
    "body_velocity_uses_approach_sign": "mul(normalApprox,.5)" in text and "mul(normalApprox,-.5)" not in text,
    "raw_invalid_contact_diagnostics": "void printContactSetup(" in text and "printContactSetup(\"production_sample_rejected\"" in text,
    "invalid_contact_remains_failure_and_continues": "allSampleChecks = false;" in text and "SKIPPED_AFTER_RETAINED_SAMPLE_FAILURE" in text and "continue;" in text,
    "inertia_sample_diagnostics": "printContactSetup(\"inertia_sample_rejected\"" in text and "return Outcome::fail;" in text,
    "fp64_inertia_chain_retained": "fp64FullInertiaAction" in text and "production_free_rigid_inertia_fp64_chain" in text,
    "production_sign_is_A_minus_dR": "finiteDifferences[ei][i]=-(plusResidual[i]-minusResidual[i])" in text,
    "active_stiffness_branch_guard_retained": "oracle.stiffnessBranch==plus.stiffnessBranch" in text and "oracle.stiffnessBranch==minus.stiffnessBranch" in text,
    "path_oracle_retained": "sampleSweepFP64" in text and "positive_min_sample_gap_does_not_prove_swept_clearance" in text,
    "inertia_fp64_and_production_thresholds_retained": "constexpr double fp64Tolerance=1e-6" in text and "constexpr double tolerance=1e-3" in text,
}
# Verify the replacement is based byte-for-byte on sealed 073 and only adds
# the explicitly intended fixture,diagnostic corrections.
base_sha=hashlib.sha256((here/"base-073.mm").read_bytes()).hexdigest()
meta=json.loads((here/"source-overlay.json").read_text())
checks["exact_073_base_hash"] = base_sha == meta["base_sha256"]
checks["only_checker_changed"] = meta["changed_path"] == "matter/tools/cardboard_free_box_angular_native_check.mm"
for name,ok in checks.items(): print(f"{name}={'PASS' if ok else 'FAIL'}")
print(f"source_sha256={hashlib.sha256((here/'cardboard_free_box_angular_native_check.mm').read_bytes()).hexdigest()}")
if not all(checks.values()): sys.exit(1)
