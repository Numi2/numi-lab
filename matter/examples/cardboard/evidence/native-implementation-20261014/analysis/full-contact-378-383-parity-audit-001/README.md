# Full-contact 378 vs 383 parity audit

This is a read-only offline audit of two completed native full-contact attempts and one separate native producer/reducer control. It does not launch native code or build software.

The audit result is `COMPARISON_COMPLETE_OUTPUT_PARITY`. Both full-contact runs completed normally without timeout or source mutation, and both exited 1. They have the same 32 Armijo records and deterministic status rows. The first nonzero solver status is Newton 9, code 10, environment 0, object 0, failing index 9, with diagnostics `[3.76308299e-6, 3.76308321e-6, 1.1920929e-7, 24]` in both. Both diagnostic transaction histories record accepted controls 1199 and 1200, then a rejected control 1201 with `native_step_exception`. The failed numerical attempt remains a failure; parity does not promote it to convergence or physical qualification.

All 145 Newton-7 center-capture arrays match byte for byte (11,670,948-byte payload). The source-physics fingerprint is unchanged (`915f50b10edc2e46`). The device-program fingerprint changes from `465b0d473f9f4973` to `1a4d455a205df1db`; the change is reported, not normalized away. It is bound to each run’s source-build and runtime artifact hashes. Each raw snapshot differs at exactly eight bytes (offsets 12–19), the serialized device-program fingerprint (`465b0d473f9f4973` vs `1a4d455a205df1db`); all remaining snapshot bytes and all transaction JSON/history hashes match. Captured contact-candidate ordering is identical at Newton 7. That capture does not establish candidate ordering at the later Newton-9 failure, so the audit makes no ordering-based failure-selection claim.

The 378 and 383 scenario commands have identical physical flags after replacing only executable/material/output paths. Both use the same source-bound Nagasawa material asset. The full-contact checker sets `localMaterialNewtonIterations = 16u` in all three compile paths; no material-iteration CLI override exists or was passed. `authored_newton_iterations=32` is the separate fixture/world Newton budget. Accurate-paper override is absent (`accurate_paper=false`); compensated material arithmetic is on.

The separate 381 native routing checker exited 0 with 169 checks, but reports `physical_steps=0` and marks its synthetic producer controls as non-qualifying. This is routing/control evidence only, not physical contact or full-box qualification.

The audit retains the exact source-107 to source-115 source delta, source-build receipts, command/prediction/receipt hashes, all 32 Armijo status rows, transaction records and binary payload comparisons, capture manifest/fingerprint differences, and 381 control result in `report.json`.

Run the auditor again with:

```sh
python3 audit.py \
  --candidate-run /Users/home/cardboard-implementation-20261014/native/deterministic-private-ccd-full-contact-383 \
  --control-381 /Users/home/cardboard-implementation-20261014/native/deterministic-private-ccd-producer-reducer-381 \
  --output /Users/home/cardboard-implementation-20261014/analysis/full-contact-378-383-parity-audit-001/report.json
```

Run CPU contract tests with `python3 -m unittest -v test_audit.py`.
