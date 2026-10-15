# Selector-5 production kernel check (source-bound input, GPU not run here)

This instrument dispatches the production selector-5 kernels for the coarse RAP projection/solve, graph patch LU factorization, local patch correction solve, and overlap gather. It is a bounded kernel arithmetic check; it is not an end-to-end FGMRES trial, a full-preconditioner qualification, or a material/box validation.

## Source and evidence boundary

The prepared numerical inputs originate from the selected source-182 Newton 23, environment 0 RAP/operator audit. That captured run was selector 3 (regional RAP only), not selector 5. The tet connectivity comes from the independently hashed `operator-audit.topology.npz`; the fixed/free mask is decoded from the selected-generation `candidate_nodes` array. The graph map is rebuilt by `buildRegionalGraphSchwarzLayout` from source snapshot 065 and compared element-for-element against an independent Python BFS using actual global node IDs. Equal counts alone do not establish topology identity.

The RAP matrix/factor and `P`/`AP` come from the source-182 selector-3 sidecar. The patch matrices are exact principal blocks extracted from the separately audited FP32 CSR operator. This bypasses native selector-5 FEM patch-matrix construction; therefore the check exercises the production coarse projection/solve and graph factor/solve/gather kernels on prepared data, but does not establish that production FEM assembly and runtime layout are correct for this generation.

The Metal library is from the `graph-schwarz-native-063` overlay, while the graph layout helper and host ABI headers are from source snapshot 065. Those are different source lineages. This is intentional for a standalone bounded kernel check and must not be described as same-build solver evidence. No GPU run or native output bundle is present in this folder.

## Reproduce the inputs and build

From this directory:

```sh
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-angular-and-graph-schwarz-integrated-065/matter/include \
  graph_layout_export.cpp -o graph_layout_export
python3 prepare_inputs.py
clang++ -std=c++20 -fobjc-arc -fmodules -Wall -Wextra -Wpedantic -Werror \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-angular-and-graph-schwarz-integrated-065/matter/include \
  selector5_graph_kernels.mm -framework Foundation -framework Metal \
  -o selector5_graph_kernels
```

The checked input bundle is `canonical182-input/selector5-kernel-input.bin`. Its manifest is `canonical182-input/bundle.json`; exact mesh-derived map and global IDs are in `canonical182-input/layout.json`. The deterministic source inputs are pinned in `prepare_inputs.py` by SHA-256.

A parent-owned native run, if scheduled, uses:

```sh
./selector5_graph_kernels \
  canonical182-input/selector5-kernel-input.bin \
  native-output.bin \
  /Users/home/cardboard-implementation-20261014/analysis/graph-schwarz-native-063/NumiMatterGraphSchwarz.metallib
python3 verify_kernel_output.py \
  canonical182-input/selector5-kernel-input.bin native-output.bin \
  --bundle-manifest canonical182-input/bundle.json \
  --metallib /Users/home/cardboard-implementation-20261014/analysis/graph-schwarz-native-063/NumiMatterGraphSchwarz.metallib \
  --report native-output-audit.json
```

The verifier binds input, bundle manifest, checker source and metallib hashes. It checks captured RAP against `PᵀAP`, coarse projection and solve, every patch LU permutation/reconstruction and correction residual, and the global overlap gather. Its default relative-error ceiling is `1e-3`; all component errors are reported. Failure or missing native output is not converted into a pass.

## CPU tests already run

```sh
python3 test_verify_kernel_output.py
python3 /Users/home/cardboard-implementation-20261014/analysis/selector5-regional-capture-reader-001/test_reader.py
```

The output-verifier tests use a hand-solvable one-node diagonal system and inject invalid-factor, bad-gather, dimension and trailing-byte errors (5 tests pass). The capture-reader tests cover byte/hash/generation contracts and explicitly reject same-count wrong global IDs under an exact topology binding (7 tests pass). These CPU tests validate the adapters, not the Metal kernels.

## Source-bound input dimensions

The prepared source-182 environment-0 bundle has 882 nodes, 1,920 tetrahedra, 846 free nodes, 36 fixed nodes, 26 RAP modes, 13 graph patches, 2,053 patch-node entries, 6,159 patch DOFs/pivots, and 4,718,637 local matrix floats. The layout checksum is `2602968157` (FNV-1a 32-bit over the same ordered arrays as runtime). Source-182 never contained a native selector-5 graph-factor sidecar; the reader in `../selector5-regional-capture-reader-001` labels captures without a same-run topology artifact binding as structural-only.
