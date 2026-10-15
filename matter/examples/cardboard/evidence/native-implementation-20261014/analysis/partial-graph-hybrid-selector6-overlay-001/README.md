# Selector 6 hybrid partial graph overlay

This source-bound overlay adds a distinct FEM preconditioner selector 6. It uses the bounded partial graph-Schwarz correction on graph-covered free nodes and the current linearization’s already-built 3x3 nodal tangent inverse on uncovered free nodes. For an uncovered row it applies Pc + B(r - A Pc), where r is the current FGMRES input vector. The new kernel reads fgmresBasis + columnOffset explicitly; it does not substitute a column of the regional basis for r.

The physical operator, right-hand side, Armijo acceptance, tolerances, certificate logic, and material recipe are unchanged. Selector 5 remains on its legacy graph builder, mask validator, patch application, and RAP behavior. Its shared matrix-builder call binds allowUncoveredFreeNodes=0; selector 6 alone binds 1 and additionally validates current block coverage.

## Bounded scope and failure behavior

Selector 6 is admitted only for one immutable explicit FEM object with 1–2 environments, at most 32,768 FEM nodes and 131,072 topology tetrahedra, no contact or rigid rows, no cohesive faces, no fields/Human/vascular/MPM, no adaptive/mutable/learned/identified material path, and compact authored regional IDs. Graph layout is at most 128 patches × 64 nodes. The graph’s packed per-environment device arenas remain capped at 96 MiB with uint32 index closure; regional P/AP/coarse/fine scratch retains the existing 256 MiB cap. The separate 3-row float4 block arena is checked as E*N*3*16 bytes and against the device’s maximum single-buffer length.

Every Newton linearization rebuilds the block rows. The builder clears all three rows/marker before assembling and marks a block valid only after finite inversion. The subsequent hybrid mask validator checks each environment’s current candidate: fixed tags 1 and 3 require zero inverse mass and need no block; free tag 0 requires positive inverse mass and either graph coverage or a marked, finite block. Other tags fail. The RAP factor kernel resets its validity word before factoring, and selector 6 turns an invalid RAP into NM_STATUS_LINEAR_SOLVER_FAILURE. Patch LU initializes validity before factoring; failed patch factors write a fatal solver status and cannot fall back. Completed Newton tails may reuse a graph factor only when the existing nonlinear-completion latch and prior valid factor permit reuse.

## Verification

CPU layout test command:

    clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror       source/matter/tools/regional_graph_schwarz_layout_check.cpp       -o /tmp/cardboard-selector6-layout-check       -Isource/matter/src -I<source-120>/matter/src
    /tmp/cardboard-selector6-layout-check

Result: graph_schwarz_layout_checks=40. The check covers deterministic partial coverage, dynamic fixed/free predicates, invalid/nonfinite blocks, invalid RAP or patch factors, malformed CSR ranges, and the two-level coarse/patch/3x3 action. The RHS differs from the first regional basis column in the test. It also exercises a synthetic 20,682-node/89,088-tet chain to check sizing only: 76,583,508 graph bytes plus 1,985,472 block bytes. It is explicitly not the actual cardboard mesh topology and is not a measured peak-RAM result.

Source syntax checks passed for the Metal library, runtime.mm, compiler_impl.cpp, validation_impl.cpp, cardboard_probe.mm, and the regional compiler-check translation unit. The Metal syntax check emitted two -Wsometimes-uninitialized warnings in the existing patch factor/refinement threadgroup control (sharedValid and refinementValid); the frozen base emits the same two warnings. The probe syntax check disabled one unrelated existing unused-function warning. No full CMake build, GPU launch, native convergence run, or actual-mesh admission test is included here.

## Next required evidence

The overlay needs a fresh integrated build, the compiler and CLI CTest cases, and a bounded native fixture that exercises both covered and uncovered free rows, then dynamic fixed/free changes across Newton generations. A failed RAP, patch LU, or invalid/missing block should be shown to fail closed. Until those checks and a successful native run on a real source-bound mesh complete, selector 6 is an experimental conditioning path and does not establish shipping-scale solver capacity or convergence.
