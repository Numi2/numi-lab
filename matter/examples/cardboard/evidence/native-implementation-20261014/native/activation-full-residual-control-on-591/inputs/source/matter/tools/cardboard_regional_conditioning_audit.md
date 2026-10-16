# Replaying a rejected regional-conditioning capture

`cardboard_regional_conditioning_audit.py` is a bounded, standard-library CPU
audit for the `numi.cardboard.regional-conditioning.v1` capture written beside
a rejected FEM step. It validates the manifest dimensions, exact raw arena
hashes, little-endian scalar layouts, factor/pivot/node-block markers, the
outer prediction/receipt/result, the required native solver options, and every
sealed source/runtime/external input hash before computing any matrix metrics.
It refuses dimensions above 4 environments, 100,000 nodes, or 32 modes and
limits each input arena to 64 MiB and all raw arenas together to 256 MiB. It
also caps the sealed input map at 10,000 entries.

Run it with the outer probe directory inferred from the capture path, or pass
`--run-dir` explicitly:

```sh
python3 matter/tools/cardboard_regional_conditioning_audit.py \
  /path/to/output/rejected-conditioning-000002 \
  --output /path/to/output/rejected-conditioning-000002/conditioning-audit.json
```

The result is explicitly `rejected_step_diagnostic_only`. For each environment
it recomputes the symmetric RAP spectrum from the captured effective basis and
`AP` in double-precision host reductions, checks the exported pivoted LU against
that reduction, and measures the Ritz `A`-invariance residual
`AP*q - lambda*effective_P*q`. It also reports how much of that residual lies
inside versus orthogonal to the captured coarse space. These are not FGMRES
right-hand-side residuals: the capture does not include the full operator,
Krylov basis, or rejected linear-system RHS, so it cannot determine the first
failing subsystem or prove that RAP caused a run rejection.

For the retained cycle-055 capture, the outer observation binds the rejected
environment-0 step to native status 10 (`NM_STATUS_NONLINEAR_SOLVER_FAILURE`)
after 256 FGMRES iterations; it records certificate residual `4.099e-5`,
correction `0.013817`, and free-force imbalance `0.4099 N`. Those values locate
the terminal transaction outcome but do not identify which internal solve,
line search, or acceptance test produced it.

The accompanying `test_cardboard_regional_conditioning_audit.py` builds a tiny
synthetic sealed capture and checks successful interpretation plus fail-closed
handling of changed hashes, dimensions, factor validity, and source binding.
