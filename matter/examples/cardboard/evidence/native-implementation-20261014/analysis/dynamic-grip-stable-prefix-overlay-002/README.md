# Dynamic-grip graph stable-prefix overlay 002

This overlay repairs the selector-5 graph layout used by dynamically changing FEM grip masks. It preserves the complete source-free graph as the exact leading patch sequence and appends only supplementary patches made from nodes that were fixed in the authored source. Consequently, acquiring a temporary grip on an initially free node continues to use its original source-free patch, while releasing an initially fixed node has a separate local patch.

The runtime builder still records the all-potential-node mask for candidate validation and fingerprinting. It now passes the authored initial-free mask to the graph-layout helper. The helper constructs the source-free partition first using the historical rule, then constructs a disjoint initial-fixed partition from fixed-to-fixed tetrahedron adjacency. The production FEM operator, residual, coarse basis, candidate constraints, accepted state, nonlinear tolerance, and graph kernels are unchanged.

## Bound source and topology evidence

The patch is based on frozen source 075 at `snapshots/source-completed-graph-tail-and-signed-contact-oracle-075` (parent `35bfb518c542a306f563cc364ef8496837068b9a`). `manifest.json` binds the frozen-source manifest and each modified base file. `patch.diff` changes only `regional_graph_schwarz.hpp`, its runtime call site, and the CPU layout checker.

For the retained 882-node, 1,920-tet source-182 topology, the frozen source-072 helper produced 13 patches, 2,053 patch-node entries, 6,159 DOFs, and 4,718,637 matrix floats. The new helper's diagnostic source-free-only mode produced a byte-identical serialized layout. The combined layout has 27 patches, 2,089 patch-node entries, 6,267 DOFs, 4,721,337 matrix floats, and 75,783,068 device bytes across two environments. The original 846 source-free nodes keep identical descriptors, node/row prefix, node coverage, and node-to-patch CSR. The appended 36 entries are all initially fixed nodes, and every initially fixed node receives coverage.

`evidence/source182-prefix-audit.json` and its input/output binaries retain the comparison. The topology input SHA-256 is `f0f5bc136f50f7e4cb3c66005316edbf47279b64a42ec531bfbe54e5189493b9`. This is a CPU graph-layout identity check, not a native solver or physical result.

## Verification

The isolated layout checker passes 27 checks:

```sh
clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I overlay/matter/src \
  overlay/matter/tools/regional_graph_schwarz_layout_check.cpp \
  -o regional_graph_schwarz_layout_check
./regional_graph_schwarz_layout_check
```

The runtime source also passes an Objective-C++ syntax-only compile with the overlay helper include directory ahead of the frozen source include directory. No Metal target was built.

The source-topology prefix audit passes using the retained binaries:

```sh
python3 audit_layout_prefix.py \
  --source-mask inputs/canonical182-source-mask.bin \
  --baseline evidence/baseline-oldheader.bin \
  --source-only evidence/baseline-newsourceonly.bin \
  --combined evidence/canonical182-source-plus-fixed.bin \
  --output evidence/source182-prefix-audit.json
```

The patch applies cleanly to the frozen 075 base and the three resulting file hashes match the overlay. No Metal build or GPU run was performed here; root must integrate and run the source-bound native guard-parity and coupon checks. This overlay makes no convergence or shipping qualification claim.
