# Tet 1186 targeted FP64 directional epsilon diagnostic

This one-row CPU diagnostic replays tet 1186 from the same rejected source101 Newton-23 capture, with the exact captured accepted history fixed for center and all five plus/minus endpoints. The independent FP64 center and endpoint projections use the v3 state-scaled criteria: scaled residual `<=2e-10` and dynamic correction `<=2e-8`, with the same four-update/24-backtrack bound. The analytical tangent is evaluated at the converged center and compared with the central stress difference along the captured local direction.

| Target `||delta F||_F` | Central FD vs tangent relative L2 | One-sided gap relative L2 | Projection / recorded branch predicates |
|---:|---:|---:|---|
| `1e-6` | `9.6744191e-8` | `8.5373237e-4` | converged / stable |
| `3e-7` | `2.0727821e-7` | `2.5571473e-4` | converged / stable |
| `1e-7` | `1.2149033e-6` | `8.4595713e-5` | converged / stable |
| `3e-8` | `1.1276622e-6` | `2.5532055e-5` | converged / stable |
| `1e-8` | `2.1807604e-6` | `5.7642304e-8` | converged / stable |

All five central comparisons were computed; no epsilon was selected as a replacement for the full-row result. Although the `1e-6` and `3e-7` steps are below the `1e-6` relative threshold, the uniform 1,728-row audit at its predeclared `1e-7` target still has its one resolved exceedance. This one-row sweep does not make the full-row comparison adaptive or pass the original common-step gate.

All five center/endpoint triplets passed the v3 projection gates and matched the recorded branch predicates (multiplier increment sign, yield-gap sign, and smoothed FB selector). The center yield gap is about `9.576e-12`, very close to zero; stability of those recorded signs is not proof that the point is far from a transition. The independent reference reports no separate discrete hardening-branch predicate. At the largest steps, the one-sided gap is large, consistent with visible local nonlinearity; at the smallest step the two one-sided estimates nearly agree while their common tangent difference is larger. These observations do not separate truncation from FP64 arithmetic effects and establish no universal noise floor.

The run retained 313 `RuntimeWarning`s from the rationalized FB residual's eager `np.where` division branch at `bilinear15_reference.py:40` (162 divide-by-zero and 151 invalid-value warnings). They were not suppressed. The final JSON was serialized with `allow_nan=False`.

No native FP32, accepted coupled solve, physical calibration, or all-direction qualification is claimed. This sweep supplements but does not alter the uniform package006 result.

Run with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 replay_tet1186.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 verify.py
```
