# P151: offline construction progress at 2,220 surface pairs

P151 records a new offline construction endpoint for the same frozen surface. The exact full check and finalizer report **2,220 surface pairs, 0 contained point/bank pairs, 0 self pairs, 185 complete banks, and 237 typed anchor rows**. Candidate SHA-256: `9b67a968dd5aebdf8d0cdf197dbd0701052ba449afa30082ff08a95aa3d28a29`. The reported maximum source displacement is 11.99648895195475 mm under the recorded 12 mm source cap. The fixed reference area floor passes at 0.100093569 against 0.1. The state file intentionally retains `candidate_selected: false`; this package reports the measured endpoint without changing that owner selection field.

This sequence continues the previously published 2,326 endpoint (P150, exact parent main `24a11b9fb4753080d9e2579d419613ea0ca432d6`). Normal full-plane round 013 checks at 2,282. A retained-component proposal produced an exact 2,294 endpoint; combining it with the normal 013 candidate gave 2,274. The composed topology chain accepted 19 flips, changed 38 faces, left coordinates bitwise unchanged from 2,274, and passed its full check at 2,232. The one-SCP component stage then produced the 2,220 endpoint. Stage receipts record prepare 35.4467 s, raw fit 50.5465 s, component exact-self step 30.5826 s, full 545 check 36.6997 s, and the finalizer. Only one raw direction capture is retained for this endpoint.

## Failed branches and diagnostics

The full-plane rounds 010, 011, and 012 are retained with their candidate arrays, exact self-pair witnesses, reports, source pins, and stage receipts. Their exact self checks failed at 113, 23, and 252 self pairs, respectively; those failed checks do not supply a containment result. Round 013 passed exact checks at 2,282 / 0 contained / 0 self. The component 2,294 result and combined 2,274 result retain their source/proposal receipts and exact reports. The compact line-search target diagnosis is included as a diagnostic only: its report identifies the 40,715 area face as a global limiter and records 113 round-010 and 23 round-011 new pairs as not fit-known SAT roundoff. The diagnosis is not a fresh geometry fit or physical evaluation.

Large fit-stage stdout mirrors are omitted only where the structured `fit/report.json` and exact stage launch/execution records remain. Their original paths, byte sizes, SHA-256 values, and reason are recorded in `omitted-payloads.json`. No raw failed endpoint candidate or exact self-pair witness is omitted.

## Sources and verification

The package includes the component-area backtracking and retained-component selection helpers, their focused tests and the fresh **10 passed** receipt, the one-SCP runner, exact launch/execution metadata, and source snapshots where their hashes match. Other source/input hashes remain recorded in the original reports and launch receipts; P150 and earlier unchanged payloads are referenced at their exact published paths in `prior-publications.json`. Independent remote-content recovery and local staging-retirement receipts for P146/P147 are included under `verification/`.

## Limits

All evidence is offline construction and exact static checking. `native_admission` is false and `accepted_physical_steps` is zero. P151 records no visual inspection, native runtime, accepted physical skin/fat transfer, FEM/contact acceptance, or clinical/anatomical qualification. It does not establish closed-skin containment or physical performance. Construction counts and geometric gates must not be read as physical qualification.
