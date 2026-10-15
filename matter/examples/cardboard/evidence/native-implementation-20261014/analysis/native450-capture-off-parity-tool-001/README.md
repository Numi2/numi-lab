# CGLS capture observer parity audit

Read-only CPU auditor for the source142 pair `bounded-cgls64-exact-state-450` (CGLS capture on) and `bounded-cgls64-capture-off-parity-452` (capture off). It requires the two runs to have exact physical probe arguments except the output directory, and to differ in exactly two variables: `NM_FEM_BASE_CGLS_CAPTURE_NEWTON` and `NM_FEM_BASE_CGLS_CAPTURE_PATH`. All other command and prediction environment settings, source/runtime input hashes, build provenance, and compiled world/package must match. It does not modify or launch runs.

After both complete receipts exist, run:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native450-capture-off-parity-tool-001/audit_capture_observer_parity.py \
  --on-run-dir /Users/home/cardboard-implementation-20261014/solver/bounded-cgls64-exact-state-450 \
  --off-run-dir /Users/home/cardboard-implementation-20261014/solver/bounded-cgls64-capture-off-parity-452 \
  --on-command /Users/home/cardboard-implementation-20261014/native-450-command.json \
  --off-command /Users/home/cardboard-implementation-20261014/native-452-command.json \
  --output /Users/home/cardboard-implementation-20261014/analysis/native450-capture-off-parity-001
```

The audit compares every file under each `output/`, the status5 payload, stdout/result, every residual-Armijo and true-residual record, and all remaining stderr after excluding only capture-only metadata and timing lines. It requires complete terminal receipts; it does not accept an accepted-prefix comparison. The CGLS binary/marker are expected only in the on run. A parity pass is observer-history evidence for this run pair only, not solver, convergence, performance, or physical qualification.
