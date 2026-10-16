# Publication 033 milestone draft: source198 contact feasibility and bounded CGLS

**Status:** draft for archive preparation; external to the publication checkout and not committed. The publication checkout is at HEAD `877cd6e69657c9a7b4e5032bcd475705bcc06c85`. Its 12 source198 working-tree files match the frozen source198 hashes; this task changed no checkout files. Source198’s own frozen metadata records parent revision `a5ccffa52ffe368b49c877094a5081b7ee246ebf` and base snapshot `source-shared-contact-trial-189-buildfix003`.

Source198 binds to `snapshots/source-contact-damped-memory-198-001/frozen-source.json` (SHA-256 `16474dca955f9a953f78cf211054c10967c8d31cd5daa8bf5d8476741ffe8cfc`) and its successful build `build-source198-001/source-build.json` (SHA-256 `8f6c95304c23b6c7149ec6b270d54eafdf4507ea26f3420e39e523c09dd500ad`). The integrated-build receipt confirms the exact source set. The build binds 816 selected source files and 1,034 support files, including the full-contact checker and `NumiMatter.metallib`. Seven targeted CPU CTests, 13 solver-environment tests, and the bounded damped-CGLS oracle pass. These are implementation checks, not physical validation.

Source198 carries the shared exact FEM trial/CCD geometry, the opt-in damped-CGLS pilot, the FEM-only deformable-contact feasibility gate, and revised shipping-memory accounting. The gate permits only exact FEM-only contact support to skip its per-feature barrier-energy decrease inequality. It still requires finite barrier evaluations and certified CCD for the exact trial endpoint; the global coupled-residual Armijo decision and final full-state certificate remain in force. MPM-only and mixed FEM/MPM supports retain their prior per-feature path. This is a narrow solver policy, not a contact-force or tolerance change.

The damped direction is experimental and disabled by default. Its selector admits only exact `NM_FEM_BASE_CGLS_DAMPING_KG=1e-3`; absent is off and malformed values reject. Damping changes the direction objective, so the prior Newton certificate is not transferred to that direction. Native helper results below exercise private nonsymmetric CGLS recurrences, not a full FEM solve.

Memory accounting avoids charging the same private arena twice and conditionally counts incidence-sort workspace only when `NM_CONTACT_NODE_INCIDENCE_SORT=1`. The checked workspace is `E * bit_ceil(6*C) * 12 bytes`; malformed settings and overflow reject. Existing topology and arena caps remain unchanged. Passing preflight arithmetic does not admit the shipping mesh: its contact/cohesive topology remains outside native admission.

## Native evidence

Native518 failed before dispatch because its helper metallib lacked `nm_fgmres_base_cgls_initialize_state`; preserve it as a setup failure. Corrected native519 and expanded native523 private-kernel checks pass at O0/O3 with identical paired stdout, but have zero physical transactions. In 523, the undamped path reports expected breakdown and the damped path reaches the requested 32/64 iteration caps; this is not convergence or a coupon result. Native524 separately passes 264 warm-start kernel assertions with zero physical transactions.

Native520 rejects the loaded contact control at Newton9 under the earlier scalar budget. Native522’s observer-off run is byte-identical to 520 across transaction artifacts, establishing observer parity for that failure. With the extended budget, scalar native525 accepts two controls, then exhausts line search at Newton18 (status10, 256 FGMRES iterations); the third transaction rolls back. That raw failure occurs before the injected postCommit fault, so it masks that negative control. Source198 scalar native527 reproduces 525 byte-for-byte across all 13 output files in each arm and has the same masked fault result.

Source198 regional-contact native526 accepts the first two controls and reaches the intended injected postCommit rejection on the third after a reported nonlinear residual of `9.63568425e-8` (threshold remains `1e-7`). It exits successfully as a checker because the injected failure is expected; that transaction is rejected and rolled back. Baseline and aliased arms have matching output hashes. This distinguishes the regional fixture path from scalar 525/527, but does not close a complete coupon.

These native contact runs use a small regression fixture (36 FEM nodes, 48 tetrahedra, one cohesive face), not the shipping assembly. No full coupon or shipping assembly claim follows. Native521 is deliberately excluded from the proposed archive inventory because it is active per the task; this note makes no claim about its outcome.

## Qualification status

All six qualification axes in the existing ledger remain open: solver correctness; numerical convergence; component calibration; analogue validation; physical assembly in simulation; and crush/impact prediction. No source198 result qualifies a completed coupon, mesh/time convergence, corrugated material calibration, held-out analogue, released shipping assembly, or crush/drop prediction.

## Proposed new archive cases

1. `source198-build-and-cpu-validation`: frozen source, support/build bindings, integrated-build receipt, CPU CTest receipts/logs, solver-environment contract, and damped oracle.
2. `damped-cgls-helper-history-518-519-523`: source190 validation plus native518 failure, native519 pass, and native523 expanded pass, retaining helper binary/metallib bindings and failure.
3. `shipping-memory-and-selector-guards`: source192 single-arena receipt and expected old-header failure; source196 incidence-sort overlay/report and CPU controls; source197 exact damping-selector preflight/receipt.
4. `contact-observer-and-baseline-failure-520-522`: native520 command/output, native522 observer-off output, and diagnostic-parity report.
5. `warm-start-kernel-contract-524`: native524 prediction, receipt, runner, and stdout.
6. `scalar-contact-budget-and-masked-fault-525`: native525 command/receipt/raw baseline+alias outputs, byte audit, and postCommit masking addendum.
7. `regional-contact-feasibility-526`: native526 command/receipt/raw baseline+alias outputs; include the sealed independent transaction audit at analysis/native526-527-independent-transaction-audit-001/report.json (SHA-256 e847266eff05ed9b42a2ed493ee06bb1e8e1560945dad0f197d4d2b301f8ec5c).
8. `source198-scalar-repeat-527`: native527 command/receipt/raw outputs, byte comparison to 525, and same masked-fault outcome.

Do not include native521 or imply that it has a closed result. Preserve the expected 518 setup failure and the 525/527 pre-postCommit failures in the archive. Exact source, report, command, receipt, and output paths and hashes are indexed in `evidence-index.json`. This draft does not modify the ledger or checkout.
