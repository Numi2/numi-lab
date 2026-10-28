# Registered O/I/D sidecar reader check

This evidence is based on repository base commit `384c24870c8d3484114dbe10cd7c61e06cb18ebf`. It checks the standalone strict reader in `apps/NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp`; no application hook, native simulation, GPU work, child tetrahedral mesh, or physical admission is included.

## Reproduction

On SSH host `numi`, from `/Users/numi/Work/skin-fat-native-integration-209`:

```sh
/usr/bin/c++ -Iapps -Iinclude -Imatter/include -std=c++23 -O1 -Wall -Wextra -Wpedantic -Werror \
  /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/P/native-child-sidecar-reader-223/reader-check.cpp \
  -o /tmp/registered-sidecar-reader-check
/usr/bin/time -p /tmp/registered-sidecar-reader-check \
  /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/volume-worker/registered-surface-sidecar-222
```

The check compiles against the unchanged public headers and exercises both pinned snapshots. Attempt-006 must reject atomically at I node 2411 under the existing source-barycentric tolerance `1e-8`; its barycentrics are `[0.5000012117859428, -1.7837966222076952e-06, 0.5000005720106794]`. Attempt-007 normalized-source-bary must parse O/I/D, donor frames, and source/parent-registration ancestry. A deliberately wrong D-face SHA-256 must reject atomically.

## Result

`registered-sidecar-reader=PASS`. Attempt-007 rows are O/I/D nodes `32218/15022/15022` and faces `63959/29999/29999`; 86 donor frames and 15,022 I plus 15,022 D parent-registration rows load. The strict tolerance was not relaxed.

Attempt-007 normalizes 4,526 I/D source-face barycentric rows, retains the raw values in the delta table, and recomputes source residuals. Maximum barycentric change is `6.132083583864742e-05`; maximum registered source-point shift is `2.2943821318902677e-06 m`; world reconstruction error is `0 m`. The parent-cell registration rows and face/node source ancestry are unchanged by that normalization.

O parent-tet registration remains unavailable for all 32,218 O nodes. The sidecar set has no production child `.node/.ele`, child volume-node frame/parent maps, or child registered-facet maps. This is reader evidence only, not a changed-child mesh or physical-admission result.

`reader-check.cpp`, stdout/stderr, `receipt.json`, and `source-hashes.txt` pin the executable check and inputs. The synthetic native-domain-217 two-tet fixture is separate and is not used as a production child mesh.
