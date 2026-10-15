# Source380 dimensionless Armijo overlay

This is a review package for a narrow local-projection repair. It changes the
line-search merit only when the existing
`NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION` policy is enabled. The
legacy material path keeps its existing raw residual Armijo comparison.

On each accurate-policy Newton iteration, the overlay freezes the current
Jacobian row scales
`s_i = sum_j(abs(J_ij) * max(abs(accepted_j), abs(candidate_j), scale_floor))`
and compares `max_i(abs(r_i) / s_i)` for the current and trial residuals.
Trial scales are not recomputed from a trial Jacobian. The accurate policy
requires strict decrease and retains the existing Armijo coefficient. The raw residual convergence gate,
scaled correction gate, iteration budget, local equations, and material recipe
are unchanged. The per-iteration row scales and scaled trial merits are written
only into otherwise unused trace padding for a 15-state material.

## Source and evidence binding

- Base snapshot: `source-private-ccd-fracture-115`
- Base `matter/src/metal/common.metalinc` SHA-256:
  `e53feb0402a461f355cf2ee5c3af89f95eb49176178e60d997d4a9cc6588a182`
- Native motivating trace: `exact-source380-material674-trace-388`
- Native trace SHA-256:
  `586d8cb6f55feaaba900ce93fa7428e47ee0d8fded300f631190f51668424579`
- Point input SHA-256:
  `7925a6eba993e6bf8b4f536352027d18c221f95c215fe73d1b20ebaec6aeb797`
- The row-scale audit is saved in `evidence/native_trace_row_merit.json`.

At trace iteration 4, row 6 sets the raw residual maximum at `4.16e-16`,
while row 12 sets the fixed-row-scaled maximum at `1.40e-10`. At iteration 14,
the corresponding values are `5.20e-17` and `4.31e-11`. This is evidence that
raw mixed-unit L-infinity merit can be dominated by a stress-like residual
after the complementarity residual has fallen. The original trace records only
raw trial maxima, so it cannot establish whether the proposed scaled merit
accepts those particular trials. A fresh native run is required for that.

## Verification and limits

Run the dependency-free CPU reference checks with:

```sh
python3 -m unittest discover -s tests -v
```

The ten tests cover row-scale calculation, row-unit invariance, frozen weights,
strict rejection of equal/worse merit, invalid inputs, and the native current
row-dominance observation, plus source checks for the retained policy and
convergence gates. One acceptance comparison uses an explicitly
synthetic trial vector because trace388 did not retain trial residual vectors.
These checks validate the merit definition only; they do not compile Metal or
qualify solver convergence. This package has no Metal compile or native
post-overlay trial receipt. The parent run must verify native policy-on behavior
and unchanged policy-off histories before integrating it.
