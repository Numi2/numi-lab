# Selector 6 fixture repair overlay 001

Base: frozen source snapshot `source-hybrid-regional-preconditioner-124`.

The 413 failure occurs while `Selector6NativeProbe::run` calls the shared `cook()` fixture, before the selector-6 GPU kernels run. The fixture selected hybrid graph Schwarz but supplied no `femPreconditionerRegionIds`; compiler admission requires exactly one compact authored region label per explicit FEM tetrahedron. The fix supplies region `0` for regional Galerkin fixtures (one region for this one-tet topology), then removes those labels from the copied scalar-diagonal control world because region labels are only legal with a regional selector. Compiler/runtime admission is unchanged.

The stateful round-trip negative case now uses selector `7`. Source124 defines selector 6 as `regionalGalerkinHybridGraphSchwarz`, and canonical validation rejects values greater than 6 through the mixed-solver-policy guard. This restores the intended out-of-range mutation test without treating a valid mode as invalid.

A `--selector6-compile-only <liner-material>` mode runs the same `cook()` fixture and package round-trip without constructing or dispatching Metal. It confirms the fixed fixture compiles under the frozen source124 compiler and preserves selector 6, the explicit FEM object, contact-disable flags, and 4-node/1-tetrahedron layout. Strict Objective-C++ syntax compilation also passes. The repaired full native probe has not been run; 413 remains a failed native attempt and selector-6 kernel/runtime parity is still unqualified.

Native412 is retained as the earlier wrong-material-argument failure (`paper material does not expose the complete 13-value accepted state`). Native413 is retained as the corrected-material invocation that reached the missing-region-ID fixture failure. Neither result is overwritten.

## Reproduction

Apply `patch.diff` with `-p1` to the exact base snapshot. `validation_commands.txt` records the source-bound compile and CPU checks used for this overlay. All compile outputs and binaries in this folder are external verification artifacts; no frozen source tree or source124 build product was overwritten.
