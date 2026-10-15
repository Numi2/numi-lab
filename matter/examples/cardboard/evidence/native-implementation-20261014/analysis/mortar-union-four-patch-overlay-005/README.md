# Exact shared-edge guard overlay 005

This is an external, opt-in implementation overlay based on the source-091 four-patch partition work and checker overlay 004. It narrowly handles a measured FP32 clipping artifact: two triangles whose mapped FP32 barycentric vertices share an exactly identical edge and whose third vertices lie on robustly opposite sides are mathematically non-overlapping, although the existing float clipper can produce a tiny positive-area sliver.

The production change leaves `clipTolerance = 1e-7f` and `kMortarPatchOverlapAreaTolerance = 1e-10f` unchanged. It still rejects nonfinite clipping, degenerate triangles, any positive-area overlap without the exact-edge proof, uncertain/near-collinear orientation, same-side triangles, endpoint mismatches, and capacity failures. The shared header predicate requires two exact matching 3D FP32 barycentric endpoints, a nonzero projected edge, and both orientation determinants to exceed a conservative binary32 error bound with opposite signs. It does not loosen the area threshold.

The guard is called only after the existing clipping result exceeds the unchanged overlap floor. The CPU control exercises source-derived inputs and negative controls. `-ffp-contract=fast` was used for the CPU check to match the source clipping arithmetic. The full-contact checker target compiled and linked, including the Metal library, but no GPU/native execution was run; this overlay is not native multi-patch qualification.

## Shipping partition audit

The source-bound shipping CPU audit over 183,033 canonical parent pairs found 34,682 pairs rejected by the source FP32 clipper as overlapping above the existing threshold. The exact-edge proof accepts 34,185 of those as strictly opposite-side triangles sharing two exactly equal mapped FP32 vertices. The remaining 497 are unproven and remain rejected. A separate wider-precision clipping cross-check found zero positive overlap above the threshold for those source-only slivers, but that observation is not used by this guard and does not justify accepting the 497.

The audit also reports maximum same-parent mortar multiplicity 4, 27 partition cells, maximum parent-area error 0.0003281831741, and 2 FP32 patch-side areas at or below the native floor. This is a CPU geometry preflight only; it does not establish runtime admission, contact accuracy, shipping execution, or full-box qualification.

## Build and checks

The self-contained patch is `exact-shared-edge-guard-005.patch`. Its base is the exact `build-source` in overlay 004, itself based on snapshot `source-rationalized-paper-and-native-checkers-091`. Apply with `patch -p1` at the source root after verifying the manifest hashes.

CPU checker:

```sh
clang++ -std=c++20 -O2 -ffp-contract=fast -Wall -Wextra -Werror \
  -I build-source/matter/include mortar_shared_edge_guard_check.cpp \
  -o mortar_shared_edge_guard_check
./mortar_shared_edge_guard_check
```

Host/Metal compile-only check:

```sh
cmake -S build-source -B build-host -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE=/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14
cmake --build build-host --target numi-matter-cardboard-full-contact-armijo-check -j 4
```

The CMake hunk adds the new shared header to the shader dependency list so a future header edit triggers a Metal rebuild. Neither command executes a GPU simulation.
