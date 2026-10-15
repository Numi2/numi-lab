# Captured center replay layout audit

## Evidence binding

- Historical native output: `native/analytic-current-failure-22-center-replay-169/center-replay.jsonl`
- Capture JSONL SHA-256: `c27d5634f77d4f431d3a0fdc5b55a03958b6c3a986942a6b1c49e097fd6e76e1`
- Receipt: exit code 2, execution started, source inputs unchanged, no timeout.
- Declared source snapshot: `snapshots/source-captured-reference-and-rigid-instrument-repair-integrated-057`
- Snapshot checker SHA-256: `78dd413ca33fc9cb056cfdf9e2886a2393ace7032ed0350b0deeaa87e871d1c9`
- Snapshot Metal checker SHA-256: `81d69f6dc4dfa7f18dc6aa12f49952218df4af35126a8e991011986b27b86241`

## Corrected layout finding

The archived source-057 checker does not have the later residual-row candidate-input layout. Its host packs F[9], direction[9], F-low[9], then the accurate-policy flag at input index 27, and passes an accepted-state vector of exactly 13 values to the production checker. The archived Metal kernel reads the same input offsets and reads accepted state from its separate buffer. Therefore the later 32-versus-13 candidate-base defect does **not** invalidate run 169's production-center P/state/tangent replay comparisons.

A separate, unexecuted WIP residual-row instrument did have a 32-slot accepted-state packing inconsistent with its Metal candidate base. That defect was corrected in the source-59 work: accepted paper state occupies 13 slots and the candidate base follows F, F-low, and Fdot at offset 40. Treat that WIP instrument's pre-fix candidate rows as invalid. It was not part of the source-057 executable bound to run 169. A fresh source-bound replay (planned run 177) is required for the new residual-row comparisons.

## Remaining limits on run 169

The legacy ratio comparing differently scaled residual norms remains invalid and is not a qualification metric. The reported two-word representation estimate is not a measured Metal noise floor; near-zero comparisons remain inconclusive without a measured floor. Run 169 exited with code 2 and remains a failed diagnostic attempt, not a physical or convergence result. Preserve its valid production-center comparisons separately from the invalid legacy ratio and from the unexecuted WIP residual-row data.
