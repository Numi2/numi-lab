# Captured 419 alternative-direction export

This package exports three Tikhonov directions and one raw Cauchy control for the exact source125 control-step 1, Newton-iteration 48 capture. It is a CPU-only linear-algebra diagnostic. It does not change the solver, run native physics, or qualify a nonlinear step.

The bound free-DOF operator is the captured unsymmetrized CSR matrix `A=M+dt^2*K` with 2,538 rows and 71,046 nonzeros. The RHS is the exact captured pre-FGMRES mechanical impulse vector. The source-bound conditioning report gives `sigma_max=1.094732845241445`; the Tikhonov parameters are `lambda/sigma_max = 0.01, 0.001, 0.0001`. Each solve minimizes `||A p-b||^2 + lambda^2 ||p||^2` using SciPy LSMR directly on `A`, without forming normal equations. All three reached LSMR stop code 2; the smallest lambda needed 54,112 iterations. The raw Cauchy control uses `g=A^T b` and `p=(g^T g)/((A g)^T(A g))*g`.

Each output direction is little-endian FP32 `[environment,node,xyzw]`, shape `[2,882,4]`. It stores the proposal in environment 0’s free xyz rows; fixed xyz rows, all w lanes, and the quiet/reference environment are zero. The paired native replay payload is copied from the exact 424 `NMFDSWP1` input and differs only in that direction-buffer byte range. Candidate, accepted node/material history, parameters, cached tangent, and tetrahedron data remain bitwise unchanged. `verify.py` independently checks artifact hashes, payload byte preservation, and CSR residual/correction/`g dot p` metrics after reading the actual FP32-exported directions back.

The rounded-vector predicted residual ratios are approximately 0.8678 for raw Cauchy, 0.2146 for `lambda/sigma_max=0.01`, 0.06605 for `0.001`, and 0.03243 for `0.0001`. The respective free correction norms are about `1.43e-4`, `3.20e-3`, `1.07e-2`, and `3.73e-2 m/s`, compared with the captured Newton correction norm of about `188.12 m/s`. These are linear predictions at one captured candidate only. A smaller CSR residual does not establish a valid Armijo trial, material update, CCD path, accepted state, or terminal certificate.

To replay a payload later, use the source125 native residual-direction executable and the exact compiled package recorded in `report.json`, for example:

```sh
/Users/home/cardboard-implementation-20261014/analysis/native-residual-direction-replay-419-001/build/native_residual_direction \
  /Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack \
  /Users/home/cardboard-implementation-20261014/analysis/captured419-regularized-directions-001/replay-inputs/tikhonov_sigma_0.001.input.bin \
  OUTPUT_DIRECTORY
```

No native replay was run while creating this package. In particular, the original captured Newton certificate does **not** certify any of these alternative directions.

## Preserved first attempt

`superseded-attempt-001/` preserves the first export, where the `lambda/sigma_max=0.0001` LSMR solve was stopped at a 20,000-iteration cap (stop code 7). It is not the final proposal set. The current package reruns that solve to stop code 2 at 54,112 iterations. The first attempt’s report and manifest are retained unchanged.
