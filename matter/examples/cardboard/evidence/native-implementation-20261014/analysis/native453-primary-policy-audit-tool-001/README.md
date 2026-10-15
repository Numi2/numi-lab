# Native453 primary-policy CGLS audit tool

This read-only analyzer adapts the source-bound native CGLS snapshot reader from `native450-cgls-candidate-audit-tool-001` to the source143 primary-policy run. It checks the exact `1:all` selector, CGLS64 capture at Newton48, true-residual configuration, source-bound completion receipt, snapshot/header/layout/status, and the per-environment primary-policy selection marker.

The analyzer compares ordinary Newton Armijo records for Newtons `[0,48)` against native450. Root-all fallback records are kept visible. The only records it omits from the exact Newton-to-Newton comparison are fallback observations with a zero CGLS candidate, a same-environment Newton acceptance or certified no-op at that iteration, and a matching unused-tail/no-private-trial status role. A nonzero candidate remains unresolved even when the status field says `unused_tail_after_selected_accept`; that status describes trailing line-search scratch and alone does not prove the whole CGLS fallback was bypassed. If the selected CGLS marker or prior Newton outcome is missing, the comparison fails closed.

At Newton48 and for every later observed primary-policy record, `primary_policy_eligible` is the direction-selection evidence. A true value means CGLS was selected for that environment and the original Newton certificate does not transfer. A false value means the original Newton direction was retained under the policy-scoped trace label. Accepted, rejected, and certified-no-op Armijo outcomes are reported separately. The analyzer does not treat an accepted Armijo trial as a solver certificate or a successful coupon.

The invocation binds the source-bound 419 operator capture. If all compiled-world, compiled-pack, root/Newton, dimension, and capture-run bindings match, the analyzer compares the captured CGLS actions on native `x` and `r` against the independently assembled CSR operator. These are directional comparisons only, not whole-matrix identity or constitutive proof. Native receipt exit codes are reported literally: a normally ended nonzero-exit solver run is an observed failure, not physical success.

## Run

After native453 has produced a receipt and complete captured files, run:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/native453-primary-policy-audit-tool-001/audit_primary_policy_snapshot.py \
  --binary /Users/home/cardboard-implementation-20261014/solver/cgls-primary-after48-control1-453/base-cgls.bin \
  --run-dir /Users/home/cardboard-implementation-20261014/solver/cgls-primary-after48-control1-453 \
  --source-snapshot /Users/home/cardboard-implementation-20261014/snapshots/source-cgls-primary-policy-143 \
  --operator-environment 0 \
  --reference-run-dir /Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419 \
  --reference-environment 0 \
  --baseline-run-dir /Users/home/cardboard-implementation-20261014/solver/bounded-cgls64-exact-state-450 \
  --output /Users/home/cardboard-implementation-20261014/analysis/native453-primary-policy-adjudication-001
```

The source-bound driver command is `/Users/home/cardboard-implementation-20261014/native-453-command.json`, SHA-256 `f4889862aa03e4600ed516aa4bf46fff48a77c34908cee33852b7f5f75814bef`. The prepared protocol is `/Users/home/cardboard-implementation-20261014/analysis/primary-cgls453-inputs-001/protocol.json`; it labels the run as prepared and does not itself establish execution or qualification.

## Validation

`test_audit_primary_policy_snapshot.py` exercises the consumer using synthetic trace records: selected CGLS vs retained Newton, rejected update vs accepted update, certified no-op, missing direction marker, strict pretarget fallback classification, and ended/unchanged receipt checks. These are parser/consumer checks, not native solver evidence.

No GPU, native simulation, or production source build is performed by this tool package.
