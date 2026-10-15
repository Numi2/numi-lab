# Native source149 right-scaled CGLS audit (463/462)

This package prepares a CPU-only post-run audit for command 463. It reads the native CGLS v2 payload, validates its `.state.json/.state.bin` sidecar, joins the Newton root/index and world/program fingerprints to the same-run `operator-capture`, then reports masks, selected/quiet environment state, block inverse errors, and captured `Ax`, residual, and preconditioned normal-residual action comparisons against the source149 operator reconstruction.

Run after the complete native receipt exists:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native-cgls-right-scale-audit-463-001/audit_capture.py \
  --case /Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463 \
  --command /Users/home/cardboard-implementation-20261014/analysis/native462-463-right-scale-command-package-001/native-463-command.json \
  --source-snapshot /Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149 \
  --expected-environment 1 \
  --output /Users/home/cardboard-implementation-20261014/analysis/native-cgls-right-scale-audit-463-001/report.json
```

The command requires complete capture markers, matching root/Newton selection, matching fingerprints, a valid state-sidecar hash table, one selected environment, and the exact block3 non-certificate marker. It compares the same-run operator to avoid assuming the prior Newton trajectory remained byte-identical. FP32 word differences, signed-zero differences, and numeric action differences are reported separately. It does not impose a new action tolerance or noise floor. It does not establish a Krylov certificate, nonlinear success, full-coupon convergence, performance gain, or physical/material qualification.

The parser unit tests exercise v2 segment layout, reject v1 and trailing bytes, and distinguish numeric equality from signed-zero word identity. They passed with the qualified Python 3.10.12 runtime. This is an audit-tool preparation package; it contains no native run result.
