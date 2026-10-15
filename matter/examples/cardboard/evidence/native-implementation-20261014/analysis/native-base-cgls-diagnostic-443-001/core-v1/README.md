# Core helper v1

Frozen base: source-coupon-cache-admission-139, `frozen-source.json` SHA-256 `e22f2d17ff26f71e1b2919db32652820c664e02802a2a8fca8067a6cccaa680e`.

The only new production source file is `source/matter/src/metal/base_cgls.metalinc` (SHA-256 `24563b7d4d54c894c1f6ace75dca060429159752500c394e78a69a4fc0ae3b2c`). Apply `core-overlay.patch` to append its include after `fgmres.metalinc`’s namespace, and `cmake-dependency.patch` to register it as a shader dependency. This is kernel code only; runtime selection, resource allocation, production A/A-transpose bindings, acceptance, and candidate installation are not included.

Binding ABI:

- `nm_fgmres_base_cgls_initialize_state`: `(E, cap, selected, scalars, control, errors)` at buffers 0–5.
- `..._initialize_vectors`: `(U, E, mask, rowEnv, selected, b, ATb, x, r, s, p, q, errors)` at 0–12.
- `..._measure_initial`: `(U, E, mask, rowEnv, b, r, s, scalars, control, errors)` at 0–9.
- `..._form_alpha`: `(U, E, mask, rowEnv, q, scalars, control, errors)` at 0–7.
- `..._update_xr`: `(U, E, mask, rowEnv, scalars, control, p, q, x, r, errors)` at 0–10.
- `..._complete_iteration`: `(E, control, errors)` at 0–2.
- `..._form_beta`: `(U, E, mask, rowEnv, s, scalars, control, errors)` at 0–7.
- `..._update_p`: `(U, E, mask, rowEnv, scalars, control, s, p, errors)` at 0–8.
- `..._finalize`: `(E, control, errors)` at 0–2.

Reductions are `E` threadgroups × 32 lanes; vector kernels use `U` threads; state, completion, and finalization use `E` threads. `ATb` may alias `s`. The helper expects `U=2*E*N`, with mechanics then zero backing rows for fields. Quiet environments require exact zero RHS/mask/ATb. The helper deliberately does not apply physical or solver tolerances.

CPU Python tests pass (6/6). Metal compilation and Objective-C++ syntax checks pass. No GPU execution is included.
