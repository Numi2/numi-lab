# Native 464 regional capture audit, version 4

This CPU-only audit joins the source-151 regional capture from native run 464 to
its exact retained state and the 419 FP64 CSR operator. The state-to-operator
join is exact for candidate/accepted nodal state, tetrahedra, accepted material
history, environment parameters, correction, RHS, tangent cache/validity,
compiled dimensions, world, time step, topology, and the free mask. The older
462/463/455 stable arrays also match, but their sidecars omit
`micro_generation`, so those historical capture generations are not proven
identical.

The source-151 selection sidecar is stale: it records `[0,0]` because it is
copied before the CGLS layout kernel writes selection. The audit preserves this
metadata defect. It derives environment 0 from same-capture Armijo selection
and the FGMRES converged state, then verifies the final CGLS binary selection
segment `[1,0]`.

The graph patch comparison retains the original per-entry absolute gate
`1e-9`; it fails. The maximum free/free entry error is `7.339122992666702e-8`
at patch 0, global DOF (1029,1029): captured `0.4552142918109894`, reference
`0.4552143652022193` (reference magnitude `0.4552143652022193`). The relative
Frobenius errors are much smaller—`5.926118342203329e-8` over all entries and
`7.4665773032055e-8` over free/free entries—but these norms do not pass or
replace the entrywise gate. The global Frobenius difference/reference norms
are `1.0123826584321797e-6 / 17.08340265874232`; free/free norms are
`1.0123826584321797e-6 / 13.558858595056005`.

Per-patch free/free Frobenius relative errors are retained for all 27 blocks;
the largest is patch 11 at `8.481713174761412e-8`. Patches 13–26 contain no
free/free entries, so their free/free relative errors are null; their fixed
identity entries are exact. AP vs CSR A*P relative L2 is
`1.0428333470875882e-7`; captured PᵀAP vs CSR PᵀAP is
`3.632648387137247e-8`; base RAP LU reconstruction vs CSR PᵀAP is
`1.2272751346105972e-7`. The base factor is marked valid. The actual
restart-augmented consuming view remains unadmitted because its consumer-cycle
generation was not captured. No native Metal action parity, convergence,
physical result, mesh scaling, or shipping qualification is claimed.

Reproduce with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_native464_v4.py \
  --case /Users/home/cardboard-implementation-20261014/solver/cgls-regional-state-capture-464 \
  --environment 0 \
  --output native464-audit-v4-reproduced.json
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v \
  test_audit_native464_v3.py test_audit_native464_v4.py
```

The interrupted scalar comparison is preserved in the earlier prep folder;
the final comparison is a vectorized equivalent of the source-151 CPU
reference. This is a bounded CPU audit, not a native action replay.
