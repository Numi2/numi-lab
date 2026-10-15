# Source-bound CPU replay of the first-control local projection

This package compares the successful 283 first-control N0 capture with the failed 288 first step. It is a CPU FP64 equation replay; it does not execute Metal and does not claim to reproduce the failed native transaction.

The 283 run captured 3,456 liner/medium tetrahedron inputs at root 0, Newton iteration 0. The replay used each row’s captured FP32-high/low deformation, accepted state, and compiled parameters, and rebuilt the authored initial seed (accepted plastic/history states plus the elastic stress update). It kept the same 15-state equations, `2e-8` authored residual threshold, line search, and `2e-8` scaled-correction criterion. Only the maximum iteration count was varied.

All 3,456 CPU rows converged at each ceiling 16, 32, and 64. Each ceiling returned bit-identical FP64 states; the maximum local iteration count was 13. The 283 native residual-only capture reports 14 as its largest iteration count, while tet 1528 reports 2 native iterations. The CPU replay of tet 1528 takes one accurate-policy iteration and agrees with the residual-only CPU root within `3.11e-14` maximum state component. The full residual-only versus scaled-policy CPU comparison differs by at most `4.15e-14` maximum state component.

The failed 288 run retained status code 5 and a rolled-back material-state file, but no rich operator capture or per-tet local projection trace. The parent’s native status report identifies tet 1528 and reports determinant `0.99751335` with zero FGMRES iterations. The corresponding successful 283 captured tet has `det(F_high + F_low) ≈ 0.997513365`, but that agreement is not a substitute for the failed run’s exact in-flight F/state/seed. The CPU results therefore show that an iteration ceiling of 16 is not sufficient to explain failure on the captured 283 inputs; the native cause remains unresolved.

The next causal check needs a native failure trace containing exact F high/low, accepted state, initial authored seed, per-iteration state/residual/correction, material flags/budget, terminal local failure, and determinant. Then compare ceilings 16/32/64 at those exact native inputs without changing the law, equations, tolerance, or line search.

Files:

- `replay.py` and `replay.json`: all-row CPU projection comparison and source bindings.
- `budget_sweep.py` and `budget_sweep.json`: 16/32/64 ceiling comparison.
- `failure-comparison.json`: run bindings, tet 1528 details, and the bounded interpretation.

The captures and native run directories are referenced read-only at `/Users/home/cardboard-implementation-20261014/solver/`; they are not copied into this package. SHA-256 identities are recorded in the JSON artifacts.
