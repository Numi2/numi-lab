# Native run 415: status-5 and 409 parity audit

This read-only audit compares `/solver/two-stage-coupon-first-descent-capture-415` against `/solver/two-stage-certified-coupon-409`. The 415 status-5 payload was decoded using the source-125 `NMFEMStatus5CaptureGPU` ABI. Both receipts report exit code 2, no timeout, and unchanged source inputs. Both results are failed after one accepted step; `physical_validation` is false.

The 35-file output trees have the same paths. Thirty-three files are byte-identical. The two differences are output manifests only: three absolute material paths point at their respective snapshots, and the rejected-conditioning manifest carries the expected different device-program fingerprint. The compiled matter package and result JSON are identical. All 64 residual Armijo records and 4096 fresh true-residual records are byte-identical. After removing only timing lines and the explicitly enabled/skipped/completed capture metadata, all remaining stderr lines match exactly; stdout is identical.

The status-5 capture is the first private trial, at control step 1 / Newton iteration 26 / trial index 0 / element tet 1464 in environment 0. It reports `detF=0.99594718`, `max|stress|=1,179,185,280`, and the captured material maximum-stress bound is `1,000,000,000`. The stress therefore exceeds the bound by 17.918528%. In source 125, `nm_fem_internal_forces` first applies `femMaterialOutput`, then records `(detF, maximumAbs(stress), 0, 0)` when `!valid || maximumAbs(stress) > material.validity.z`. The record has no independent `valid` bit, so it is not evidence that local projection succeeded; it does show that the candidate violates the stress maximum. Newton 26 later accepted alpha `0.0078125` (1/128), so this is a rejected alpha-1 private candidate, not the accepted state or the terminal Newton-48 failure.

There is also a capture-lane defect: runtime passed `femLineSearch`, whose `.x` field is not the selected candidate alpha, to the status-5 kernel. The kernel wrote `.x` into the field labeled alpha, yielding zero while setting the “available” bit. The private alpha lives in `femArmijoMetrics.z`. The separate overlay `status5-alpha-lane-overlay-001` corrects only this observer binding and read. It does not change mechanics or acceptance and has not been GPU-qualified.

The coupled full-state capture requested at Newton 48 was skipped because the fixture has two environments and that capture path only admits one. This audit does not attribute the later Newton-48 failure to the Newton-26 trial, and it does not establish convergence or physical coupon qualification.

Reproduce the parity report with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_415.py
```

The full machine-readable result, including per-file hashes and the exact JSON manifest differences, is in `audit.json`. `decoded-status5.json` preserves the ABI decoder output. The source-125 source manifest and the relevant runtime/kernel files are SHA-bound in `manifest.json`.
