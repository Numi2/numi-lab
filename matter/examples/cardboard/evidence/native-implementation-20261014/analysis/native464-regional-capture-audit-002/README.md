# Native 464 base regional graph audit

This CPU-only audit joins the source-151 regional capture from native run 464
against the exact 419 FP64 CSR operator and retained state sidecars. It checks
that the selected environment and state are bound, compares patch blocks,
AP, and the unaugmented base RAP, and explicitly refuses to claim that the
restart-augmented map was consumed.

The 464 terminal receipt is exit 2 after an executed, non-timeout run; captured
inputs are unchanged. The state-to-419 join is exact for accepted/candidate
nodes, accepted tetrahedra, material history, environment parameters, Newton
correction, residual RHS, tangent cache and validity, plus dimensions, world,
dt, topology, and free mask. Historical 462/463/455 stable arrays also match
exactly, but their sidecars omit `micro_generation`, so exact generation
identity across those older captures is unavailable.

The selector sidecar is defective: it reports `[0,0]`, but source-151 copies
that buffer before the CGLS layout kernel writes it. The audit preserves these
words as stale metadata. It derives environment 0 from same-capture
`armijo_metrics.w == 3` and the FGMRES converged flag, then cross-checks the
final CGLS binary's selected segment `[1,0]`.

Against the bound 419 CSR, the captured graph patch free/free maximum absolute
difference is `7.339122992666702e-8`, above the unchanged `1e-9` audit bound;
fixed identity/cross entries are exact. Captured `AP` versus CSR `A*P` relative
L2 error is `1.0428333470875882e-7`. Captured `PᵀAP` versus CSR `PᵀAP` is
`3.632648387137247e-8` relative L2, and base LU reconstruction versus CSR
`PᵀAP` is `1.2272751346105972e-7`. The base factor-valid flag is set. The
request for the actual augmented consuming view is refused because its
consumer-cycle generation was not captured.

Run with the qualified interpreter:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_native464_v3.py \
  --case /Users/home/cardboard-implementation-20261014/solver/cgls-regional-state-capture-464 \
  --environment 0 \
  --output native464-audit-v3.json
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_audit_native464_v3.py
```

The scalar patch comparison from the earlier attempt was interrupted after
3m+ and about 1.1 GiB RSS; that partial attempt is preserved in the prep
folder. The final v3 audit uses a vectorized equivalent and completes in
seconds. This is a bounded CPU reference audit, not a Metal action replay,
solver convergence result, physical result, or shipping qualification.
