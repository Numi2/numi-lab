# Selector-5 graph-patch refinement checker

This standalone checker exercises the production Metal factor, solve, and apply kernels with the iterative-refinement ABI frozen in source 071. It dispatches both `refinementMode=0` (unrefined control) and `refinementMode=1` against the same source matrix, LU factorization, pivots, coarse coefficients, and RHS. Separate negative/guard dispatches cover an exact zero RHS in both modes, the completed-cycle latch, an invalid factor, an unsupported mode, and a singular principal matrix. The checker retains the output bundle before applying the FP64 audit. The selector-5 solve shader and shared ABI hashes are unchanged from source 071; this directory contains checker revision 004, compiled against source 071's frozen snapshot.

For zero-RHS solve/apply controls, the checker distinguishes numeric nonzero and nonfinite outputs from IEEE signed zero, and reports negative-zero counts separately for solve and apply. It does not rewrite the retained output. Completed-cycle and invalid/failure outputs retain their stricter bitwise-zero checks. The Python verifier independently recomputes each patch's `b-Ax` using FP64 scalar `math.fsum` row products and checks the unchanged `1e-3` relative residual gate. It also requires at least one mode-0 patch to fail that gate and mode 1 to improve every such patch. A failed arithmetic gate returns exit 2 and keeps the output/report for diagnosis.

Build the host checker against the exact frozen source-071 ABI headers:

```sh
xcrun --sdk macosx clang++ -std=c++17 -fobjc-arc \
  -framework Foundation -framework Metal \
  -I /Users/home/cardboard-implementation-20261014/snapshots/source-refined-graph-angular-residual-instrument-071/matter/include \
  selector5_refinement_kernels.mm -o selector5_refinement_kernels
```

Run only after a metallib containing the source-070 `nm_fgmres_graph_*` kernels has been freshly built. Use the real-mesh prepared input from the previous kernel instrument:

```sh
./selector5_refinement_kernels \
  /Users/home/cardboard-implementation-20261014/analysis/selector5-native-kernel-check-002/canonical182-input/selector5-kernel-input.bin \
  native-refinement-output.bin \
  /path/from/the/source-bound/build/NumiMatterGraphSchwarz.metallib

python3 -W error verify_refinement_output.py \
  /Users/home/cardboard-implementation-20261014/analysis/selector5-native-kernel-check-002/canonical182-input/selector5-kernel-input.bin \
  native-refinement-output.bin --report native-refinement-audit.json
```

The input bundle is bound by `bundle.json` and is derived from the selected source-182 FEM operator and actual connectivity, with selector-5 patches rebuilt from that mesh. This is a prepared-kernel experiment: it does not execute a selector-5 FEM/FGMRES transaction, establish convergence of the shipping coupon, or validate any cardboard material or physical claim. The historic source-067 mode-0 failure, source-067 metallib symbol failure, and native221 checker failure remain retained in their original evidence. The old executable copied from checker003 is preserved with a `.source003-stale` suffix and is not authoritative for this checker revision.

The small Python tests use a hand-built identity patch solely to test the verifier and its failure paths. They are not native kernel evidence:

```sh
python3 -W error test_verify_refinement_output.py
```
