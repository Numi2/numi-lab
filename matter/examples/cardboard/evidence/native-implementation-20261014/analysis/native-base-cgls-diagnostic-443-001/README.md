# Native CGLS stage diagnostics for source 141

This checker-only overlay adds `CGLS_STAGE` records around each existing direct-kernel dispatch. Each record prints both environments' control words, scalar words, error bits, and x/r/s/p/q mechanical norms. The records bracket initialization, both matrix actions, initial measurement, each alpha/update/completion/beta/update stage, and finalization. This is intended to identify the first stage that changes a status or error flag. It does not alter the CGLS helper or its inputs, the test matrix, dispatch geometry, acceptance assertions, production source, or solver policy.

The retained source-141 checker failure is `selected environment did not finish with a valid CGLS status` at `/Users/home/cardboard-implementation-20261014/native/base-cgls-direct-kernel-443/`. The sealed helper has a suspicious predicate in `nm_fgmres_base_cgls_measure_initial` that treats nonzero active-mask rows as constrained; this package does not repair it. Parent-owned native execution should establish the observed status/error transition before any source fix.

## Binding

- Base source snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-bounded-native-cgls-141`.
- The checker is based on the source-139 direct checker, with the same `base_cgls.metalinc` SHA-256 recorded in `source-overlay.json`.
- Only `native/base_cgls_native_check.mm` differs from the source-139 direct-checker package. `native/diagnostic-checker.patch` is that exact diff.
- `native/compile_only.sh` passed (Metal library build and strict Objective-C++ syntax check). This is compile evidence only, not native checker evidence.
- The original 443 failure and the original checker are retained unmodified.
