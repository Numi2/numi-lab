# Native 525 independent byte audit

This package records a read-only audit of source-bound native 525 outputs.

To reproduce the audit on the same host, run python3 audit_snapshots.py. The script binds the command spec and receipt, verifies the source frozen-map hash and all five build-artifact hashes, checks the material asset hash, then hashes every file in the baseline and alias directories. It requires identical path sets and bytes between those runs and compares every actual file size/SHA against the native receipt. It also checks the transaction JSON accepted/rejected/rollback semantics and required checker stdout lines.

The audited output has 13 files in each run: 3 transaction JSON files, 6 raw before/after snapshots, and 4 logs. All 13 baseline/alias pairs match exactly and all 26 outputs match the receipt. Controls 1 and 2 are accepted. The third transaction is rejected before its trial because of pretrial status code 10 at index 18 after 256 FGMRES iterations; its accepted-vector hashes and control cursor roll back to control 2.

The checker reports 5,976 assertions and 6 native FEM transactions across baseline and alias runs. The run uses the synthetic fixture. It demonstrates alias parity and bounded transaction behavior only; it is not full coupon, shipping, or physical qualification.
