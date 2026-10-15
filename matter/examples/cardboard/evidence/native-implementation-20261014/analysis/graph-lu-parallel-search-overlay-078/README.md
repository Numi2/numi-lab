# Graph-patch LU search parallelization overlay 078

This is an external, source-bound optimization experiment based on frozen source snapshot `source-stable-free-graph-and-general-arithmetic-076` (parent revision `35bfb518c542a306f563cc364ef8496837068b9a`). It changes only the selector-5 graph-patch LU scale/pivot searches and the host binding needed for an explicit mode word. The shared checkout, CMake, and GPU runtime were not changed or run.

The kernel already dispatches one 256-lane threadgroup per patch. In search mode `1`, lanes scan disjoint matrix entries for per-lane scale maxima and disjoint pivot-column rows for local pivot candidates. Lane zero combines only the 256 local summaries; exact ties select the smallest global row index, matching the old sequential `>` search's first-row rule. A row swap distributes nonoverlapping columns across lanes and keeps the pivot-vector swap on lane zero. The legacy branch remains mode `0` and is the runtime default. For nonfinite input/column scans, mode `1` repeats the old ordered scan to retain its partial diagnostic. The per-entry lower-column division and trailing Schur update section is byte-identical to frozen 076.

The prepared native checker compares both search modes on the retained source-derived canonical-182 matrix bundle, saves both factor outputs before its parity gate, and requires bitwise equality of LU storage, pivots, factor-valid flags, original-matrix copies, and status. It is prepared, not executed: no Metal device run or speed result is claimed. The bundle is a real source-182 mesh/operator capture, while the graph layout was rebuilt by the source-065 helper; this is an isolated preconditioner-kernel input, not same-build selector-5 FEM qualification.

## Verification performed

- CPU FP32 mirror: 7 tests pass. It compares legacy and parallel summaries/factor outputs on deterministic finite matrices (sizes 3–96) and controls exact pivot ties, zero and singular matrices, NaN/Inf with partial-scale diagnostics, the strict pivot-floor boundary, and trailing-update overflow.
- Metal compile-only: source overlay built with Metal 4.0, `-O3`, `-fno-fast-math`; metallib link succeeded. Two `-Wsometimes-uninitialized` diagnostics also occur when compiling the unmodified frozen source and are unrelated existing threadgroup-initialization warnings.
- Objective-C++ checker: syntax-only check and host link succeeded against frozen-076 headers.
- `source-overlay.patch` and `checker/extend-prepared-input-parity.patch` both pass isolated `git apply --check`; applied file hashes match the packaged overlay.
- No GPU execution, factor parity result, runtime performance measurement, or full-coupon result is available yet.

## Reproduction

From this directory, run the CPU controls with:

```sh
python3 tests/test_lu_search_parity.py
```

The checker source is `checker/selector5_refinement_kernels.mm`; its CLI remains:

```sh
selector5_refinement_kernels inputs/selector5-kernel-input.bin OUT.bin NumiMatterGraphSchwarz.metallib
```

It writes `OUT.bin.factor-search-parity.bin` before asserting equality. Build the metallib from the overlay's `overlay/matter/src/matter_kernels.metal` with the frozen-076 include roots and the same `metal4.0 -O3 -fno-fast-math` flags recorded in `evidence/metal-compile-command.txt`. Build the checker with C++20, Foundation, Metal, and frozen-076 `matter/include`. Do not use the frozen metallib: it lacks buffer 13 for the new mode selector.
