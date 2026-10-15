# Bilinear-15 Fischer–Burmeister value replay and narrow patch

This external diagnostic package is based on the immutable source snapshot
`source-direct-projection-diagnostic-089`. It contains an opt-in compiler
rewrite for one recognized Fischer–Burmeister value in the source-bound
15-state bilinear Hill material, plus CPU replay evidence. No shared checkout,
Metal runtime, GPU test, material parameter, tolerance, authored Jacobian,
or local stopping rule is changed here.

For the positive-sum branch, the patch evaluates the algebraically equivalent
value

`(epsilon^2 - 2*a*b) / (sqrt(a^2 + b^2 + epsilon^2) + a + b)`.

The branch selector retains the original authored expression when `a+b < 0`.
The accurate 15-state layout is admitted only with a positive `epsilon`; a
denominator floor protects eager evaluation of the unselected expression.
Residual directional derivatives and candidate-state Jacobians are still
generated from the unchanged authored expression, whose recognized FB
derivative already uses stable analytic coefficients. The default material
program and the 13-state/default path do not select this rewrite.

The source-bound CPU verifier is:

```sh
python3 validate_fb_rationalization.py
```

It checks the captured native296 row subset, verifies the native294 selected
trace's deformation words against run283, rebuilds the compensated C++ replay,
and checks its direct FB values against all 16 retained native trace rows. It
also checks the real-arithmetic identity and first derivatives at 90 decimal
digits on cancellation, branch, and regular control inputs. The generated
`cpu-validation.json` records all input and source hashes.

## Evidence and limits

Native296 found six accurate-policy local projection failures at environment 0,
tetrahedra 567–569 and 1527–1529. All six are the same captured local input;
the four neighboring control tets 566, 570, 1526, and 1530 pass. The default
policy passes all ten selected rows. Native294 records a representative
16-iteration local solve at tet1528. A CPU implementation using the production
two-word compensated primitives reproduces the direct residual values recorded
in every trace iteration exactly. On the positive-sum failure state the direct
residual is cancellation-sensitive, while the rationalized value remains
finite. This identifies a numerical value-evaluation issue in the captured
state.

The corrected value formula is not yet compiled into a native Metal build or
run through the six native points. Its effect on the unchanged scaled-correction
gate is unknown. The representative native294 solve still had a scaled
correction of `7.92197383e-8` against the unchanged `2e-8` criterion at its
stalled iterations, so this value repair alone is not shown to resolve the
projection failure. There is no complete nonlinear solve, calibration,
physical validation, or shipping-box qualification in this package.

The exact run288 candidate was not captured. Its observation proves that the
requested local iteration count was 32 while the compiled material budget was
16; see [the erratum](ERRATUM-288-compiled-budget.md). Run283 remains a useful
captured control/input set, not a substitute for run288's candidate.

The patch is proposed for parent review and a fresh source-bound native test.
All failures and earlier artifacts remain preserved.

An independent FP64 comparison of native296, recorded in
`analysis/retained-paper-allpoint-comparison-296-002/report.json`, adds an
important limit: among valid accurate-policy rows, 26 medium-material tangent
direction norms exceed the separate `1e-3` comparison threshold (worst
`0.0020605` at environment 0 / tet1634). Stress rows in that comparison remain
below `1e-3`; 1,728 zero-direction rows are not relative-error evidence. This
pointwise result is not explained by the liner FB value patch and remains an
independent accuracy issue. The comparison report's classification explicitly
limits it to one captured direction per point, not a componentwise or complete
trajectory qualification.
