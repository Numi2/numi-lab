# Local material-projection tolerance sensitivity patch

This scratch-only patch tests whether local generic-material projection accuracy contributes to the global operator/FD mismatch. It changes exactly two acceptance comparisons in `projectMaterialState` from `2.0e-5f * (1 + stateNorm)` to `2.0e-6f * (1 + stateNorm)`. The unmodified source and both original values remain untouched; there is no macro or default change.

Patch: `projection-tolerance-2e-6.patch` (SHA-256 `c61d75d2c1628241f96e9fd2aa57d8c5e611e1020085707513c6955d106cc0fc`). It passed `git apply --check` against `/Users/home/numi-cardboard-20261006` and was not applied.

## Coverage

The two and only two generic local projection convergence checks in `matter/src/metal/common.metalinc` are the Newton early-convergence check at lines 1237 and the final post-cap check at line 1321. Generic stress evaluations route through `projectMaterialState` at line 1356; the generic tangent path does likewise at line 1486. The calls at lines 1647 and 1680 are one-sided fallback tangent probes for stateful generic materials without implicit state; the Haj-Ali assets take the earlier implicit-tangent branch and do not execute that fallback. The external operator-FD +/- residual evaluations call the stress path, so they will use the tighter criterion through line 1356. Thus the alternate source image changes projection acceptance consistently for the production force and tangent calls relevant to this asset, without claiming the fallback probes run here.

The specialized plasticity branch returns before these generic Newton comparisons, so the patch affects generic implicit material programs such as the Haj-Ali Hill assets, not specialized von Mises/Drucker-Prager projections. It does not alter the material law, `q_floor`, `fb_epsilon`, the local Newton iteration cap, global tolerance, accepted-history transfer, or any ABI.

## Alternate Metal image command

The existing `matter/CMakeLists.txt` custom command compiles Metal with `-std=metal4.0 -O3 -fno-fast-math`, including `matter/include` and the repository `include`; it does not forward CMake C++ flags to `xcrun metal`. Apply the patch only to an isolated frozen source copy, then from that copy's root build a separate image with the same shader flags and include paths:

```sh
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
  -I "<frozen035-source>/matter/include" \
  -I "<frozen035-source>/include" \
  -c "<frozen035-source>/matter/src/matter_kernels.metal" \
  -o "<out>/NumiMatter-localtol-2e-6.air"
xcrun -sdk macosx metallib \
  "<out>/NumiMatter-localtol-2e-6.air" \
  -o "<out>/NumiMatter-localtol-2e-6.metallib"
```

This is the CMake shader compile command with only a separate output path; no define flag is needed because the scratch source contains the explicit replacement. Use the probe's existing `--metallib <out>/NumiMatter-localtol-2e-6.metallib` replay override and retain/hash that image with the exact original inputs. No build or execution was performed for this review.

## Interpretation limits

This is a numerical sensitivity experiment. A tighter criterion may improve force consistency, or it may hit the existing 16-iteration material budget / FP32 residual floor and reject previously valid projections. Retain all projection failure diagnostics and compare the same accepted snapshot, compiled pack, direction, timestep, and plus/minus perturbations. A changed global result would show tolerance sensitivity only; it would not establish that the old projection noise caused the mismatch, validate the Hill law, or provide physical cardboard validation. The accepted-history yield audit is separate and cannot identify the actual Newton-3 or perturbed local states.
