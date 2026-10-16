# Source173 compensated FP64 reference package

This package contains an additive, R-ready tool layout for the explicitly
selected compensated FP64 reference. Its equation base remains the frozen
source173 `cardboard_bilinear_fp64_reference.py`; no existing API or source file
is changed. The wrapper requires callers to pass that equation module's path,
so the helper cannot silently bind to a developer's analysis directory.

The portable files under `r-ready/matter/tools/` are the files added by
`portable-integration.patch`:

- `cardboard_dd_fp64.py`: two-word IEEE binary64 arithmetic primitives.
- `cardboard_compensated_fp64_base.py`: compensated Hill-flow helpers and an
  explicit source-module loader.
- `cardboard_compensated_fp64_reference.py`: v2 residual, Jacobian, projection,
  and implicit-tangent wrapper. It fixes the mixed direct/quotient FB scatter
  and keeps compensated q/flow terms through the residual/Jacobian path.

Existing consumers remain unchanged. A consumer opts in by importing
`cardboard_compensated_fp64_reference`, loading the current source equation
module through `cardboard_compensated_fp64_base.import_reference(path)`, and
constructing `Reference(base)`. Strict qualification still requires the
captured residual and dynamic-correction gates in the caller; `Reference.project`
retains the base helper's ordinary projection stopping rule.

The reference uses only binary64 words and compensated operations; it is not a
native Metal implementation, calibration, or physical qualification. The test
fixture is a retained captured tet used only for CPU equation checks. The full
nine-basis finite-difference audit is a separate CPU result and does not qualify
native endpoint accuracy or the full coupon trajectory.

## Reproduction

Run the focused source-bound tests with a Python environment containing NumPy
and mpmath:

```sh
python3 -m unittest -v test_referencev2_source173 test_r_ready_relative_imports
```

The nine-basis audit command and its exact source/capture hashes are recorded
in `run-018-source173-nine-basis-eps8-26/README.md` and its manifest. It uses
the unchanged gates: scaled residual `2e-12`, dynamic scaled correction
`2e-11`, and relative-L2 derivative `1e-6`. Branch-crossing endpoints are
retained and excluded from smooth centered comparisons; each one-sided result
is recorded separately. Zero realized endpoint deltas and zero reference
actions are inconclusive. Deterministic FP64 repeatability is not reported as
an absolute noise floor.
