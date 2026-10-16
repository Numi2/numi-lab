# Source175 buildfix003 sparse-scan native checker command

This package prepares (but does not dispatch) the standalone native Metal differential checker for the hierarchical sparse private-contact scan integrated in Source175. It also binds the separate full-contact Armijo checker binary/source by hash, but does not choose a material input or dispatch that physical coupon checker.

The exact target is `numi-matter-cardboard-sparse-private-scan-native-check`; it has no required arguments for the native path. `--cpu-only` runs only the host contracts and explicitly does not exercise Metal. The native run uses the build-bound `NumiMatter.metallib` path compiled into the checker.

The checker contains 49 built-in cases, including block-boundary tails, exact capacity boundaries, alternating and deterministic-random patterns, a two-environment prefailed partial-map case, and the full 6,335,568-pair shipping domain with a bounded selected set. It compares legacy and hierarchical outputs. This is kernel-level differential evidence only; it does not run the physical full-contact coupon or establish shipping qualification.

`runner.py` is intended to be copied with this package to the macmini path recorded in `command.json`. Running it without flags only verifies the package, all 806 source files, build metadata, checker, metallib, process state, and disk headroom. It dispatches the GPU checker only with the explicit `--execute` option. It blocks when any active `numi-matter-*` or checker process is present, requires 2 GiB available before reserving 128 MiB, retains a 512 MiB abort floor, and writes every execution under a fresh attempt directory.

No native execution is included in this package. Root owns the GPU queue and will run the sealed command after the active native477 job terminates.
