# Native CGLS candidate audit tool for exact run 450

This external CPU-only analyzer is prepared for the source142 same-protocol rerun. It does not run Metal or edit the solver. The native CGLS image must bind to the exact run receipt, source-bound prediction, compiled world, CGLS capture/direction markers, and 1:48 selector. Optional 419 comparison is only same-world/root/Newton directional action checks on native `x` and `r`; it does not establish whole-matrix identity.

The required `--baseline-run-dir` binds the feature-off baseline at `solver/cgls-disabled-history-parity-445`. The analyzer compares each parsed residual-Armijo record for the selected root over Newton indices `[0,48)` exactly and separately records compiled-world and compiled-package equality. It does not infer hidden state equality from trace-line parity. The selected CGLS trial is analyzed independently and cannot inherit the original Newton certificate. The existing assembled-vector 1% criterion is reported on finite, nonzero reference actions only; near-zero magnitudes are reported for root adjudication without an invented noise floor.

A `fem_coupled_capture_skipped` line is parsed by actual root/Newton and retained with its runtime reason and observed scope. No run number is hardcoded. Missing completion/skip markers stay unknown.

Run after native450 has a complete `receipt.json`, `prediction.json`, `stderr.log`, `output/compiled.nmatterpack`, `output/manifest.json`, and `base-cgls.bin`:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native450-cgls-candidate-audit-tool-001/audit_cgls_snapshot.py \
  --binary /Users/home/cardboard-implementation-20261014/solver/bounded-cgls64-exact-state-450/base-cgls.bin \
  --run-dir /Users/home/cardboard-implementation-20261014/solver/bounded-cgls64-exact-state-450 \
  --source-snapshot /Users/home/cardboard-implementation-20261014/snapshots/source-bounded-native-cgls-mask-repair-142 \
  --operator-environment 0 \
  --reference-run-dir /Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419 \
  --reference-environment 0 \
  --baseline-run-dir /Users/home/cardboard-implementation-20261014/solver/cgls-disabled-history-parity-445 \
  --output /Users/home/cardboard-implementation-20261014/analysis/native450-cgls-candidate-audit-001
```

`test_audit_cgls_snapshot.py` covers binary layout, bad padding/trailing bytes, exact pretarget trace parity/mismatch, generic capture-skip reporting, and the existing 1% action threshold including a zero-reference inconclusive case.

**Evidence status:** tool and CLI are CPU-tested only. Run450 is the sole native evidence target and its output is pending. A passing analyzer is not a completed coupon, physical assembly, or calibrated box qualification.
