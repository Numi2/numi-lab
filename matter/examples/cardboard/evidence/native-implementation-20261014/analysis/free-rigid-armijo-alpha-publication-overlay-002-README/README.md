# FEM / free-rigid Armijo alpha publication overlay

Base: frozen source snapshot `source-armijo-applied-alpha-diagnostic-integrated-066` (parent revision `5fd4486ea2bcbe903782509a769ce3935b790e55`). This is an isolated review overlay; it does not modify the snapshot or apply any new tolerance, force, CCD, or acceptance rule.

## Change

The full-contact FEM residual Armijo decision writes its accepted object alpha to `environmentLineSearch[environment]` in the same successful kernel branch that records `femLineSearch[slot].x`. A certified no-op writes `1.0` after zeroing the correction. The two runtime dispatches bind this new kernel argument. Failed trials do not publish an alpha.

The direct native regression mode `--armijo-alpha-publication` drives the production begin/measure/decide kernels through four non-descending residual trials and a fifth accepted trial at `alpha=1/16`. It then invokes the production FEM and free-rigid solution-application kernels and requires both increments to equal `0.25` for a correction of `4`. A second pass checks the certified no-op publishes alpha one, zeros both correction blocks, and leaves both candidates unchanged. This is a kernel wiring regression, not a physical simulation or convergence qualification.

The diagnosis is source-bound to native capture 203: at Newton iteration 5, FEM selected `0.0625` while environment alpha remained `1`; at iterations 6 and 7 the selected FEM alpha was `0.0009765625` and `0.000244140625` while environment alpha remained `1`. The run still failed with residual `1.42294965e-7` against `1e-7`. This patch addresses the measured FEM/rigid alpha mismatch only; a rerun is needed to determine its effect on convergence.

## Validation performed

- Objective-C++ strict syntax check passed with `NUMI_CARDBOARD_METAL_WORLD=1`, C++20, `-Werror -Wall -Wextra -Wpedantic`.
- The integrated Matter Metal source compiled to AIR from the overlay with Metal 4.0, `-O3 -fno-fast-math`; one pre-existing `sharedValid` warning in `fgmres.metalinc` remains.
- Native kernel regression was not run here; root owns the GPU execution.
- Patch applies cleanly in `patch --dry-run -p1` against the frozen 066 snapshot.

Run the targeted native regression after integrating/building the overlay:

```sh
numi-matter-cardboard-full-contact-armijo-check BULK.nmatter --armijo-alpha-publication
```
