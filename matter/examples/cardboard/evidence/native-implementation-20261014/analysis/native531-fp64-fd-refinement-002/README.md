# Native531 selected-direction FP64 epsilon refinement

This is a CPU-only refinement of the ten rows left unresolved by the immutable native531 strict FP64 layer replay. It does not run the native solver, change source, or alter any tolerance. The original replay at `analysis/native531-strict-fp64-replay-001/layers` remains untouched.

## Result

The parent tested epsilon powers 8, 10, ºw^~)Þv, 26. Its rule requires two adjacent declared epsilon grid points whose centered FP64 stress secant agrees with the implicit tangent to relative L2 <= 1e-6, with both endpoints passing the strict projection checks and retaining the center branch predicates. The sparse even-power grid left ten rows without an adjacent pair.

This refinement tested every integer power 10 through 24, using the same accepted histories, captured center deformation gradients, and captured per-tet direction. All 1,728 paper rows now have an adjacent same-branch pair on the refined grid. Each of the ten original misses has one or more adjacent pairs:

| Tet | Material index | Passing adjacent powers | Minimum relative L2 on refined grid |
|---:|---:|---|---:|
| 136 | 0 | 1:éÝyø§yÓ20 | 6.30e-7 |
| 498 | 0 | 12éÝyø§yÓ14 | 3.98e-7 |
| 499 | 0 | 12éÝyø§yÓ14 | 2.64e-7 |
| 510 | 0 | 12éÝyø§yÓ15 | 2.78e-7 |
| 511 | 0 | 12éÝyø§yÓ15 | 2.07e-7 |
| 650 | 1 | 1:éÝyø§yÓ20 | 4.48e-7 |
| 1012 | 0 | 1;§uçâçS21 | 2.56e-7 |
| 1362 | 0 | 13ºw^~)Þw15 | 3.19e-7 |
| 1363 | 0 | 12éÝyø§yÓ15 | 3.33e-7 |
| 1523 | 0 | 1;§uçâçS21 (also 23 alone) | 5.08e-8 |

At every passing point for these ten rows, both plus and minus endpoints are marked smooth and branch-preserving by the reader. Thus branch crossing does not explain the original ten misses. Their passing windows lie between old grid samples; the original two-power spacing skipped neighboring points in those windows. At smaller epsilon, most rows show the expected truncation-dominated decrease followed by a rise or oscillation in the secant error. That pattern is consistent with endpoint/projection cancellation becoming significant, but this package does not measure a numerical noise floor and does not attribute the rise to a quantified floor.

## Method and limits

The sealed reader reran all five captured state layers and the centered ideal-endpoint FP64 sweep with powers 10ºw^~)Þw24. For each epsilon, it projects centerF +/- epsilon * dt * captured_direction from the exact captured accepted history, then compares the centered stress secant against the implicit tangent at the projected center. The strict compensated-FP64 projection/refinement and branch checks are unchanged. The complete arrays, including endpoint errors and branch flags for all 1,728 rows, are in `layers/arrays.npz`; the complete reader report is in `layers/report.json`.

This resolves the ten grid-dependent misses for this one captured direction at this FP64 reference layer. It is not a measured native FP32 noise floor, a native derivative test, a 9-basis derivative audit, or evidence for full solver or physical qualification. The independent reader and source equations remain binary64/DD compensated reference calculations, not an MP oracle.

## Bindings and execution

- Parent report SHA-256: `de9603063717a6f5864e9e65e7ca6c8c933b6c42de6a84fdf7922495ad9f3761`
- Parent arrays SHA-256: `038551564ac89f298148426da9b5e6259e40fd256f091e63aa8a7d57f0f7cc65`
- Parent layer manifest SHA-256: `1db4342cdf9b866cd8280209a86cdc7cb70cd712ddb8ea68b2d42b0eb89253da`
- Native capture SHA-256: `c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2`
- Native prediction SHA-256: `b1c333332ccd527f97160af8545b5ab041620c72823291a616880ed9277dd743`
- Reader SHA-256: `45a986d399afc3c36770ef65eb4a2bf9abbef138813e9353d4c223089863252b`
- Interpreter: Python 3.13.14; NumPy 2.2.5; OMP/OpenBLAS/VECLIB/MKL threads set to 1.

The exact command and all imported reference-tool hashes are recorded in `run.command.json` and `reference-tools.json`. The first hash-pinned invocation was refused before calculation because its script contained a transcription error in the parent manifest digest; that failed preflight, the correction, and the initial write-attempt problems are recorded in `attempt-history.json`. No result from a failed attempt is used.
