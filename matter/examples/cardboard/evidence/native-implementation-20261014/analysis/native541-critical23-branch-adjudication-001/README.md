# Native 541 branch-indicator adjudication

This CPU-only addendum reviews all 138 native rows in the 23-pair critical material replay. It binds the exact native JSONL, point input, strict FP64 reference rows, compiled material package, helper binary, source-capture digest, and the source-205 material/kernel files in manifest.json.

The first adjudication report at analysis/native541-critical23-adjudication-001/report.json used a strict_reference_branch_mismatches field name for a tuple containing signs of multiplier increment, yield residual y, and a reference denominator cutoff. The field name overstated what was measured. This report preserves those observations but calls them indicators and checks them against the smooth material equations.

The Haj-Ali medium law has positive fb_epsilon and positive hardening_smoothing_width; its yield complementarity residual and tensile-hardening weights are smooth. The law does not switch to a different constitutive equation when y changes sign. The native compiled bytecode evaluates the authored smooth implicit residual. The FP64 reference uses a denominator > 1e-14 selection to evaluate a rationalized residual, and uses sign-conditioned algebraically equivalent Jacobian formulas to avoid cancellation; those are reference arithmetic choices. For the 12 indicator disagreements (six rows each at tets 852 and 1744), both native and FP64 have positive dl, the same denominator path, matching normal stress signs, and |y|/fb_epsilon < 0.032. Each method keeps its own indicators unchanged across center and endpoints.

All 138 native point records report stress, tangent, and tangent-projection valid. The maximum native-vs-strict-FP64 tangent norm error is 2.9615e-4 for this selected set; this is not an all-directions or full-paper qualification. All 23 duplicate center rows are bit-identical, which measures repeatability only.

The 46 endpoint pairs use two selected FP64-qualified powers for each of 23 tet/direction pairs. Thirty native signals exceed the sum of the endpoint discrepancies against the strict reference; sixteen do not. Among those 30, 20 centered native FD errors exceed 1e-3 relative to the strict tangent, and all 30 exceed 1e-6. No native finite-difference gate is closed. The strict FP64 realized endpoint secants are a separate reference self-check and do not establish a native arithmetic noise floor.

## Reproduce

On the source-bound macmini host with the listed files present:

    /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 analysis/native541-critical23-branch-adjudication-001/adjudicate.py
    /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 analysis/native541-critical23-branch-adjudication-001/add_fd_summary.py

Both scripts perform CPU-only parsing/reference analysis and do not dispatch Metal or mutate solver inputs. The second script adds summary statistics from the previously sealed 541 endpoint report. Final report.json SHA-256 is recorded in manifest.json.

## Limits

This is a selected point replay, not calibration, a full nine-direction native FD sweep, an absolute FP32 noise measurement, or coupon qualification. The endpoint discrepancy sum is a bounded comparison with this strict reference, not a universal noise model. It leaves the full material derivative and physical convergence gates open.
