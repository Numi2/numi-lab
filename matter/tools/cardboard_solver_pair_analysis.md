# Pair-aware FEM material-capture replay

`cardboard_solver_pair_analyzer.py` audits a rich `fem-newton-capture.v1`
record with an independent FP64 Hill-paper reference and a replay of the exact
transformed scalar residual bytecode. It was added to investigate native
Newton captures without changing the solver or its authored material law.

The replay is deliberately split in two. The analyzer validates the four
builder switches sealed in the native prediction. The compiled C++ tool
parses the retained liner and medium `.nmatter` files, recompiles their 13
residual roots with the authored row scaling, and evaluates the transformed
compensated scalar bytecode at the captured candidate. The captured candidate
already reflects native projection and initializer policies; the independent
FP64 reference solves the same residual zero-set from accepted history using
each environment's captured FP32 material parameters. It evaluates stress
and tangents at both the reprojected state and the captured candidate. Thus
candidate tangent comparison and reprojected-reference tangent comparison
are not conflated.

The input TSV preserves each FP32 high and low word independently for local
deformation `F` and the projected candidate state. The replay widens both
words to FP64 and then sums them. Accepted state, local direction, and each
environment parameter are captured FP32 values widened to FP64. It never
substitutes a packaged material default for a captured parameter row.

## Run

Build `cardboard_solver_pair_replay.cpp` as a CPU executable linked to
`numi_matter_compiler`, with include directories for `matter/include`, the
repository `include`, and `matter/tools`. Then run:

```sh
python3 matter/tools/cardboard_solver_pair_analyzer.py \
  /path/to/capture.jsonl \
  /path/to/prediction.json \
  /path/to/receipt.json \
  /path/to/numi-matter-cardboard-solver-pair-replay \
  /path/to/new-analysis-directory
```

The analyzer requires a completed native run receipt, the capture's terminal
integrity record, a verified prediction, an exact root/iteration capture
selector, and all required transform flags. It verifies source hashes against
the retained source snapshot for both material files, the probe, projection
scaler, compensated VM, compiler, and FEM material evaluation sources. The
output directory must not already exist. It contains the exact replay TSV,
JSONL output, a per-tet audit, and a summary bound by SHA-256.
Exit code `0` is reserved for a future explicitly qualified strict gate;
`3` means replay completed but the strict gate is inconclusive, `4` means
replay rows or output parsing failed, and `2` means input or source admission
failed. A timed-out native run is still admissible for a complete selected
capture if its terminal integrity and unchanged-input checks pass, but the
summary marks `full_history_complete` false.

## Interpretation

The report gives source-bytecode residual parity, FP64 candidate residuals,
state/stress/tangent comparisons, local trial-yield branch counts, realized
finite-difference directions, and branch-crossing/near-yield counts. The
capture's first projection diagnostic is the reported residual; the second
is state norm. Candidate `q` near one is described as an active yield surface,
while an elastic predictor near one is kept as an unresolved branch diagnostic.

Relative errors are descriptive. The report has no measured numerical-noise
floor, so its strict tangent gate is always inconclusive. Zero-reference
comparisons remain unresolved, and branch-crossing or near-yield finite
differences are excluded from smooth-branch comparisons. A source-bound CPU
replay does not qualify a native solver, a material model, a box simulation,
or physical cardboard behavior.

The Newton selector identifies an intermediate operator-material evaluation
inside a solver step. It is not itself a step-failure classification; use the
native run observations and receipt to establish whether that step was later
accepted.

Run the failure-injection tests with:

```sh
python3 matter/tools/test_cardboard_solver_pair_analyzer.py
```
