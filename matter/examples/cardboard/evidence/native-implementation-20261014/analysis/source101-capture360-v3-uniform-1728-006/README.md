# Source101 uniform v3 directional audit, one epsilon

This CPU-only package reruns **all 1,728 active directional rows** from the same source101 Newton-23 capture at `||delta F||_F = 1e-7`. It applies the v3 center-and-endpoint convergence requirement uniformly: source-scaled residual at most `2e-10` and dynamic state-scaled Newton correction at most `2e-8`, with at most four Newton updates and 24 residual-decrease backtracks. The captured accepted 15-state history is fixed for the center and both endpoints. The tangent is evaluated at the v3-converged center candidate using that same history and the captured direction.

The earlier package005 cannot simply be combined with a v3 run of the other 1,694 rows. Only two of its 34 rows were recomputed under the v3 correction gate; the remaining 32 were carried forward unchanged from package004. They do not establish the v3 center/endpoint gate. This package therefore reruns every row. Package004/005 are unchanged and retained.

The uniform one-epsilon result is **not a complete 1e-6 pass**. All 1,728 center and endpoint projections passed the v3 gates, and all recorded branch predicates matched across each row's center/plus/minus states. At this particular direction and perturbation size, 1,727/1,728 rows meet the rowwise relative tangent/central-FD threshold `1e-6`. Tet 1186 is the sole exceedance at `1.214903271565475e-6`; it is resolved and branch-stable, so this is a measured threshold miss rather than an inconclusive projection. No follow-up epsilon sweep was performed in this package.

The recorded branch signature covers multiplier-increment sign, yield-gap sign, and the smoothed elastic/plastic FB selector. Continuous Hill `q` and hardening factor are recorded, but the independent reference exposes no separate discrete hardening branch predicate. The result applies to one rejected coupled Newton-23 capture and one captured direction. It estimates no universal arithmetic noise floor and does not qualify native FP32 arithmetic, accepted coupled solves, other directions, physical material behavior, or a full solver trajectory.

The reference emitted 34,520 `RuntimeWarning`s from its rationalized FB residual at `bilinear15_reference.py:40` (`np.where` eagerly evaluates its division branch even when the fallback branch is selected). They are retained in `output-006/report.json`, not suppressed. Serialization used `allow_nan=False`; the completed row file contains finite JSON numbers. This records the warning behavior of this run and does not assert a general harmless-warning result.

## Package005 incompatibility and retained preliminary run

`package005-compatibility.md` records why package005's 34 rows cannot be spliced into the 1,694-row result under a common v3 gate. `output-005` is the first complete 1,728-row calculation; its row file is byte-identical to the corrected rerun in `output-006`, but its report incorrectly described the active tet IDs as a contiguous `192..1919` range. The actual IDs are `0..863` and `960..1823`, exactly the directional table domain. The corrected `output-006/report.json` fixes that metadata and is the canonical report. Both raw runs remain hash-bound.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 replay_uniform_v3.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 verify.py
```

The runner refuses to overwrite its output directory. It imports the source-bound v3 continuation implementation from package005, but computes every row itself. No production source, build, or GPU was changed or used.
