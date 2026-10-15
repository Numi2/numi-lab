# Selector-5 graph Schwarz isolated overlay

This overlay is based on the frozen source snapshot `/Users/home/cardboard-implementation-20261014/snapshots/source-phase-correct-rigid-bridge-and-implicit-reference-integrated-063`. It adds the opt-in `regional-galerkin-schwarz` selector (enum value 5), bounded graph layout, exact local principal-matrix assembly from the production FEM tangent, pivoted local factors, additive local solves, probe manifest wiring, and a source-level layout regression. Selector 5 is restricted to contact-free, one-object explicit FEM coupons (<=4096 nodes, <=32768 tetrahedra, <=4 environments); the probe rejects shipping-box use. No physical run or convergence qualification is claimed.

The source182 layout check passed 11 checks with 13 patches: eight 256-node patches and five singleton patches, 2053 patch nodes, 6159 scalar DOFs/pivots, and 4,718,637 matrix floats. The computed device arena total for two environments is 37,986,116 bytes, including the one-time uploaded metadata. Per environment, the dense matrix factors are 18,874,548 bytes; the rest is pivots, patch correction, and factor-valid flags. The host allocation now uses environment-scaled counts exactly once.

The native capture extends `numi.matter.regional_rap_capture.v1`; selector-5 capture stores only env0's post-LU patch matrices, pivots, validity, and immutable graph maps alongside the same-generation candidate, material state, deformation words, tangent cache, P/AP/RAP and status. It uses the existing aggregate 32 MiB capture cap and reports a skip when the cap is exceeded. This verifies available data for independent factor replay; it does not yet capture the first FGMRES RHS/output, so it does not establish production graph-action parity or conditioning improvement.

The exact layout source SHA256 is `a1c7a9ad8aa98b606fa300c8c8cf5de8dd4a4641d88423d2cdf68aacae1f3adc` and header SHA256 is `f8973ed2c95367e82f202be77b4f8763231dae88d17cd80c60c64ccd127391a8`. Compile and run the deterministic host check:

```sh
c++ -std=c++20 -Wall -Wextra -Wpedantic \
  -o /tmp/regional_graph_schwarz_layout_check \
  /Users/home/cardboard-implementation-20261014/analysis/graph-schwarz-native-063/overlay/matter/tools/regional_graph_schwarz_layout_check.cpp
/tmp/regional_graph_schwarz_layout_check
```

Expected output begins `graph_schwarz_layout_checks=11` and reports the counts above. Full source hashes and compiler artifact hashes are in `source-manifest.json`; `selector5.patch` is the unified patch against frozen063.

Metal translation-unit compile and library link passed on macOS using frozen063 `matter/src/matter_kernels.metal` and overlay include/source directories before the frozen063 directories. No GPU kernel execution was run. Objective-C++/C++ syntax-only checks passed for compiler, validation, runtime, and both probe build variants.
