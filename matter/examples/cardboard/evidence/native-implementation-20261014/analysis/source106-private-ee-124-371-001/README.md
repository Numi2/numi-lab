# Source106 private-contact EE witness audit

This CPU-only package adjudicates the source106 N7 last-trial contact failure at primitive pair 24/127. It does not change the solver or contact code and does not claim a native Metal re-execution.

## Native attribution

The bound source106 N7 log records trial status `code=6`, `failing_index=124`, diagnostics `[24,127,1,0]`, and trial alpha `1.1920928955078125e-7`. Candidate slot 124 is `[24,127,124,1]`; both source primitives have triangle kind 0. The cooked cohesive incidence ranges for primitives 24 and 127 each have count zero. `nm_contact_narrowphase_deformable` therefore takes the generic triangle-pair path, not `findMortarComplementContact`. Its EE loop reports `[candidateA,candidateB,edgeA,edgeB]`, so the failed subcall is edge A 1, edge B 0. The generic VT diagnostic encodes `w=3+direction`, independently ruling out the earlier VT interpretation.

## Captured ray and replay

The script reconstructs the source106 trial node positions and velocities from its N7 center, accepted state, correction, timestep, and the recorded line-search alpha, following `nm_fem_apply_solution`. It cross-checks that reconstructed state against source103 trial358. Source103 is only a cross-build geometry check: positions differ by at most one FP32 ULP and velocities match bitwise. Its actual N7 failure is a different pair, 92/127 edge 0/0, so its device capture is not mislabeled as the source106 failure.

The source103 and source106 `edgeEdgeCCD` function bodies are byte-identical. A source-ordered CPU FP32 transcription on the reconstructed source106 ray exhausts a 16-call comparison bound at iteration 15, with gap `+5.00222086e-11 m` (about `+0.050 nm`) and time `0.9577351213`. It enters the effective thickness band on call 17, iteration 16, with gap `-6.36646291e-11 m` (about `-0.064 nm`) at time `0.9580711722`; a 33-call diagnostic trace has the same CPU result. The actual source106 Metal trial returns uncertified at the 17-call production limit. This CPU/native outcome mismatch is retained and is not evidence that raising the production limit is safe or sufficient.

An independent FP64 closest-segment sweep places the minimum distance at `9.999283199365255e-6 m` near normalized time 1.0. Against the effective thickness `1.0001907156498291e-5 m`, the clearance is `-2.623957133e-9 m`. This places the reconstructed path inside the CCD thickness band near the endpoint; it does not prove a production certificate or characterize a physical material collision. The native failure means the conservative-advance routine could not certify this last trial, not that the line-search trial was proven physically infeasible.

## Boundaries

- The branch, pair identity, source code, captured inputs, and raw native status are hash-bound in `manifest.json` and `reproduction.json`.
- CPU FP32 arithmetic is a source transcription, not bitwise Metal arithmetic. The final classification margin is sub-nanometer and the native status disagrees with the CPU transcription.
- The pair has no cohesive patch references; mortar complement processing is not involved.
- The earlier VT and mortar hypotheses are explicitly superseded. No production change, GPU run, physical qualification, or iteration-cap recommendation is included.

## Reproduction

Run with NumPy available:

```sh
python3 reproduce_ee.py
```

The command rewrites `reproduction.json`. The result used for this package is checked by the hashes in `manifest.json`.
