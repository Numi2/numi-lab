# Native right-scaled CGLS checker (source overlay 149-002)

This package adds a standalone native checker for the opt-in 3×3 block right-scaling path. It compiles the actual `base_cgls.metalinc` kernels from source overlay 149-002 and is intended for the parent task to execute on the serialized GPU queue.

The checker covers both `S` and `Sᵀ` with nonsymmetric blocks containing off-diagonal entries; selected and inactive environments; fixed, quiet, field, and float4 padding rows; ignored nonfinite scale data in an unselected environment; and rejection of a selected nonfinite inverse block. It then runs the production CGLS recurrence with `B=A*S`, `Bᵀ=Sᵀ*Aᵀ`, converts the result back with `x=S*y`, checks against an independent CPU recurrence, and verifies reduction of the unscaled `||b-Ax||`.

The package also preserves the existing identity-scale checker. Its two new initializer arguments are bound to `useRightScale=0` and a valid dummy buffer, so it can be run as the unchanged identity-path control. Neither native checker was executed while preparing this package; compilation and linking are the only local validation here.

Run the CPU/compiler-only preparation with:

```sh
/Users/home/cardboard-implementation-20261014/analysis/native-cgls-right-scale-checker-149-001/native/build_only.sh
```

This compiles the helper Metal source, builds the metallib and both host binaries, and performs strict Objective-C++ syntax checks. It does not execute either binary. The root task owns the native execution.

## Source binding

The final selected production source overlay is `native-cgls-block-right-preconditioning-overlay-149-002`, based on snapshot `source-cgls-post-release-relaxation-145`. Its manifest SHA-256 is `e158080398d47e1974e34ffcd71af5cd7e563d0fd08c4eb4e53482e7f9a94906`; its patch SHA-256 is `e23ed54a067f2f2ad822d84172d21df225aa112e28307f0cbe6299a175da8a86`. The checker retains exact source copies and hashes in `manifest.json`. The 149-001 source-overlay package is superseded because its inverse-diagonal builder output was temporarily bound to the production FGMRES preconditioner buffer; 149-002 directs it to private right-scale scratch.

The kernel implementation is source-bound to `base_cgls.metalinc` SHA-256 `127d05d6323085ad6abd160ff560b95008981e9421541e9677dddbb3fa253aec`. Runtime selector, allocation, and production bindings are recorded for provenance but are not built into this standalone test metallib.

## Evidence boundary

Compilation does not prove native kernel parity, complete CGLS correctness on production FEM data, physical convergence, or any full-coupon outcome. The checker uses a small synthetic nonsymmetric matrix and does not validate production inverse-block construction from a native FEM tangent. Those remain separate native tests. The build-only receipt is in `build-only-receipt.json`.
