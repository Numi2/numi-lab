# Selector 6 native checker overlay 002

This overlay extends the existing cardboard FGMRES preconditioner checker from
source overlay 001. It adds direct production-kernel checks and a small Runtime
acceptance comparison for selectors 5 and 6; it does not change solver or
material source. Apply `source-overlay.patch` after overlay 001 to the named
base files. `parent-overlay-001.json` binds the prerequisite source delta.

The direct GPU fixture is compiled from the checker’s existing one-tetrahedron
paper fixture. The compiled topology has four FEM nodes and one tetrahedron.
Two graph patches cover nodes 0 and 1; nodes 2 and 3 are initially fixed. The
test releases node 2 in the candidate mask while node 3 stays fixed, so node 2
uses the uncovered-node block path. It checks a symmetric positive-definite
3x3 inverse block with nonzero off-diagonal terms, an RHS distinct from basis
mode P0, the covered patch correction, and zero output on the fixed node. It
also checks rejection of a missing inverse-block marker and a nonfinite block,
invalid RAP factor fail-closed behavior, and singular patch-LU factor/solve
fail-closed behavior.

The selector 5/6 Runtime comparison compiles the same physical fixture twice,
normalizing only the preconditioner selector for byte comparison. Each run
applies an external load and a release command, requires an accepted solver
certificate, a nonzero reported relative correction, and representable motion
of the loaded node. This demonstrates that the test did not pass through a
zero-correction initial latch. RuntimeStateSnapshot does not expose the exact
FGMRES Arnoldi iteration count, so the checker does not claim or print one.
The checker reports selector-to-selector position and velocity deltas; these
are diagnostics, not a calibrated error bound.

The host layout report is exact for this tiny compiled fixture only: two
patches, four flattened patch-node entries, node coverage `2,2,0,0`, graph arena
988 bytes, and uncovered block arena 192 bytes. It also prints the device's
`maxBufferLength` and Runtime's reported resident-byte accounting. Neither
quantity measures peak GPU residency. The package explicitly reports
`exact_shipping_topology=false` and `peak_resident_memory=not_measured`; this
fixture does not validate the 20,682-node / 89,088-tetrahedron shipping layout,
full-memory peak, convergence, or physical cardboard behavior.

## Validation performed

No GPU check was run in this overlay. The following source and CPU checks pass:

```sh
xcrun --sdk macosx clang++ -std=c++20 -fobjc-arc -fsyntax-only \
  -Wall -Wextra -Wpedantic -Werror \
  -I analysis/partial-graph-hybrid-selector6-overlay-001/source/matter/include \
  -I snapshots/source-padded-trial-summary-120/matter/include \
  -I snapshots/source-padded-trial-summary-120/matter/tools \
  -I analysis/partial-graph-hybrid-selector6-native-check-overlay-002/source/matter/src \
  -DNUMI_MATTER_METALLIB=\"/tmp/NumiMatter.metallib\" \
  analysis/partial-graph-hybrid-selector6-native-check-overlay-002/source/matter/tools/cardboard_fgmres_preconditioner_gpu_check.mm

xcrun --sdk macosx clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  analysis/partial-graph-hybrid-selector6-native-check-overlay-002/tools/selector6_layout_fixture_check.cpp \
  -o /tmp/selector6_layout_fixture_check
/tmp/selector6_layout_fixture_check
```

The first command is Objective-C++ syntax-only validation; it does not link or
run Metal. The second reports:

```text
selector6_exact_fixture_layout=pass nodes=4 tets=1 patches=2 flat_patch_nodes=4 coverage=2,2,0,0 graph_bytes=988 block_bytes=192 exact_shipping_topology=false peak_measured=false
```

The source patch dry-run and apply were also checked against the exact source
overlay 001 CMake file and source-120 checker; applying it reproduced both
overlay file hashes. A full source build and native checker run remain pending
for the parent worker.
