# Native 174 material replay adjudication

Run `20261016T040030Z-f41839901e414db49c077be20a95edf6` on source snapshot `source-checkpoint-warm-breakdown-173` exited 0 in 62.652s. The remote 44-file, 188,729,390-byte inventory matches every local path, size and SHA-256: **True**.

This was a GPU material-function replay with zero physical transactions. Its own receipt says `material_qualification=false`; this report does not promote it to a coupon or full-solver qualification.

## Same-source replay and captured operator

The five-state replay contains 9600 rows (center plus four captured endpoint states). For paper material indices 0 and 1, direct stress and projected state match the corresponding capture exactly; the glue cohort is reported separately because it has no 15-state paper projector.
The center nine-basis check contains 17280 actions. The paper rows match the captured same-state tangent columns exactly; this is parity against the stored operator, not an independent derivative test.
The center captured-direction comparison has 1920 rows. The row fields record tangent validity separately from projection applicability. Its end-summary `tangent_valid_rows=1728` counts `tangent_valid && projected`, so it must not be read as 192 failed tangent rows. Maximum captured-action relative L2 is 0.000119906059; maximum absolute L2 is 141129.1. Large absolute values can accompany small relative errors when the unnormalized captured direction H is large.

## Independent references and finite differences

The selected reference table has 950 paper rows; 170 also have MP fallback references. Selected paper stress relative L2 ranges to 4.30453168e-08. Selected tangent relative L2 against the packaged FP64 references reaches 1.53913667e-06 (tet 1317, material 0, state s2, column 2); that is above 1e-6 but below the separate 1e-3 material action scale. For the 170 MP-backed selected rows, maximum tangent relative error is 5.76712421e-08. The separate 17-root MP fallback package reports 17/17 high-precision fallback gates passed and a maximum 70/90/110-digit root spread of 1.15e-70; it also retains 11 native-vs-MP branch-predicate mismatches and the original 17 FP64 failures. MP fallback is a distinct reference, not an FP64 projection pass.
The epsilon sweep contains 297 pairs: 288 paper pairs are both endpoint-valid and 9 non-paper pairs are invalid (material indices [2]). On valid paper pairs, the table in `report.json` shows median/max centered-secant errors by epsilon. The max remains above 1e-3 at every tested step; no resolved finite-difference plateau or independent absolute noise floor is established. Split-float endpoint realization is reported separately and does not itself establish an accuracy floor.
Repeated identical evaluations have 0 mismatched output words across 1105 words. This is repeatability evidence only, not an accuracy or truncation floor.

The row-level source outputs, all failed/invalid statuses, and all 44 file hashes remain bound in the output inventory and the companion JSON. The overall result is **diagnostic replay complete, material qualification not established**.
