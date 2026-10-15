# Source-bound all-paper-tet tangent audit (capture 216)

## Verdict

The captured source-069 Newton 23 state passes the independent center-point tangent check for every nonzero paper direction represented in this capture. A high-precision, independently transcribed return-map replay also resolves all five targets that the ordinary-double finite-difference sweep had left without two adjacent passing refinements. The other 1,723 active paper tetrahedra already passed that adjacent-refinement rule in the original sweep. The 1,728 rows in environment 1 have exactly zero deformation direction; their directional finite-difference and assembled-direction gates remain **inconclusive**, not passes.

This is a material/operator diagnostic on a rejected transaction: native run receipt exit code 2, zero accepted steps. It is not a successful native solve, a physical-material validation, or a box-performance result.

## Full capture coverage

Capture selection: environment 0, root 3, Newton iteration 23, `dt = 9.999999747378752e-05 s`, 882 nodes and 1,920 tetrahedra per environment. Each environment has 1,152 liner, 576 medium, and 192 glue tetrahedra. The replay consumed all 24,192 recorded per-tet rows and emitted 3,456 paper-tet summaries.

Across all paper rows, captured candidate-state component error against independent FP64 reprojection is at most `9.074946e-10`. For the 1,728 nonzero-direction paper rows, native directional tangent vs. independent FP64 tangent has maximum relative error `5.813327e-8`; element tangent-force mapping (including the captured material frame) has maximum relative error `9.135297e-7`. These are below the instrument's `1e-3` tangent criterion. The result is limited to the captured local state and direction.

For the nonzero environment-0 direction, independently assembled free-node `A` action differs from mass minus the independent element tangent action by `1.590636e-7` relative, below the `1e-2` diagnostic gate. This is one captured action, not a global solve residual. Environment 1 has an all-zero direction and zero action, so its relative comparison is undefined and not passed.

## Five previously unresolved ordinary-double FD rows

The original double-precision endpoint-projection sweep did not find two adjacent same-branch refinements below `1e-6` for these rows. I reran them with an independently transcribed 13-state ideal-Hill projection and implicit tangent at 80 decimal digits, retaining the captured FP32 parameter words, `F` high/low words, accepted history, and direction. A 50-vs-80-digit repeat gives center-stress differences below `3e-43 Pa` and tangent differences below `3e-37 Pa/s` for all five. The table gives two adjacent same-branch centered-FD passes; `max|dF|` is the dimensionless endpoint perturbation, and the listed `epsilon` is `max|dF| / max|H|` in seconds.

| Tet | Center trial q | Same-branch adjacent `max|dF|` | Centered relative FD errors | Notes |
|---:|---:|---:|---:|---|
| 611 | 0.9939559705 | `3e-7`, `1e-7` | `8.7502e-8`, `9.5707e-9` | Elastic at center and both endpoints |
| 776 | 1.0002293966 | `3e-10`, `1e-10` | `2.7611e-7`, `3.0677e-8` | Near yield; all tested endpoints at these steps stay on the plastic side |
| 1569 | 0.9939086416 | `3e-7`, `1e-7` | `8.4962e-8`, `9.2949e-9` | Elastic at center and both endpoints |
| 1603 | 0.9931133081 | `3e-6`, `1e-6` | `2.3341e-7`, `2.2044e-8` | Elastic at center and both endpoints |
| 1675 | 0.9942585472 | `3e-7`, `1e-7` | `2.3136e-7`, `2.4819e-8` | Elastic at center and both endpoints |

At the ordinary-double sweep's own best step, the high-precision relative errors are respectively `5.69e-12`, `7.45e-5`, `5.67e-9`, `2.12e-9`, and `9.43e-10`. The t776 same-step result is still above threshold because its centered difference has appreciable near-yield curvature; reducing the step while remaining on the same plastic branch brings it below threshold. For the other four, the high-precision result is much lower than the ordinary-double sweep's minimum, indicating a finite-precision endpoint-projection/FD floor at those very small perturbations. The evidence does not isolate a single floating-point operation as the cause.

The independent high-precision implementation is a separate script, not a change to the constitutive law or native code. It evaluates the same captured accepted state and does not promote the rejected native candidate to an accepted state.

## Reproduction

Run with Python 3.10, NumPy 2.2.5, and mpmath 1.3.0:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/all-paper-tet-replay-232/adaptive_mp_five.py
```

The script writes `adaptive-fd-five.json` in the same directory. The original all-tet replay and ordinary-double FD outputs remain unmodified under `analysis/all-paper-tet-replay-231/output/`.

## Bindings

See `manifest.json` for SHA-256 bindings of the capture, prediction/receipt, native source-build record and binaries, all-tet replay inputs/outputs, source reference headers, and this high-precision rerun. The current checkout has advanced beyond source 069; the capture remains bound to the archived source-069 build. The build record notes that compiler system headers and OS/driver are not archived.
