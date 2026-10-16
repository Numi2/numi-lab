# Selector-5 base graph right scaling for CGLS (source155 overlay)

This is an isolated, default-off source overlay against frozen source149. It adds `NM_FEM_BASE_CGLS_RIGHT_SCALE=regional5` to the existing bounded pure-FEM CGLS direction path. The option requires the existing CGLS fallback, its 32- or 64-iteration scratch, a current selector-5 regional map, no restart augmentation, and `rigidGeneralizedCapacity == 0`. Any unsupported selector/scope, absent array, factor-validity failure, bad pivot, or nonfinite RAP solve rejects the request or latches the private CGLS error. The ordinary FGMRES path and default CGLS path keep their prior behavior.

At one Newton linearization, the forward map is the unaugmented source selector-5 formula

`S = P R^-1 P^T + H (I - A P R^-1 P^T)`,

where `H = G D`, `G` averages overlapping patch contributions, and each local map is `D = 2F - F A_local F` using the captured local LU and its one-step refinement. The added kernels implement algebraic `S^T` by reversing each factor and gather/scatter operation; they do not assume symmetric local matrices or RAP. CGLS then acts on `A S`, applies `S^T A^T` to residuals, and installs `S y` as the physical correction. All CGLS errors remain private and block direction installation; the final unscaled physical residual/certificate is unchanged.

The implementation reuses the existing CGLS vectors, right-scale temporary, selector-5 graph correction storage, and same-Newton regional arrays. It adds no persistent/private Metal allocation. It does add regional graph/RAP dispatches: forward `S` is five kernel dispatches and transpose `S^T` is six, so a K-iteration run invokes 11(K+1) map dispatches in addition to A applications and recurrence kernels. A transpose patch threadgroup has four 768-float arrays (12 KiB of declared threadgroup scratch), matching the source patch bound of at most 256 nodes. No runtime performance result is available.

## Validation

The focused Python check exercises a source-derived CPU reference with two overlapping nonsymmetric patches, nonidentity local LU pivoting, fixed-coordinate masking, a pivoted nonsymmetric RAP transpose, invalid-factor rejection, and explicit rejection of augmented/actual-consumed claims. It also checks source contracts for the opt-in parser, same-Newton base view, selected-only CGLS latch bypass, binary-capture rejection, and private RAP error latching. This does not execute the edited Metal kernels.

Metal syntax compilation of the overlay main and Objective-C++ syntax compilation of `runtime.mm` both pass. The Metal compile emits two warnings in the pre-existing source149 graph LU/refinement code; compiling frozen149 itself emits the same two warnings. No native/GPU run or runtime trajectory test was performed.

The independent source464 CPU study is a negative conditioning result for this candidate: on its fixed captured `A,b` and 64-step cap, the source151 base graph map ended at relative true residual 0.57905 in FP64 (0.92340 in CPU-FP32 arithmetic), compared with 0.22967 (0.23562) for identity and 0.08845 (0.08979) for 3x3 block scaling. The captured patch matrix also missed the predeclared per-entry `1e-9` comparison gate (maximum absolute difference `7.3391e-8`), despite small global relative matrix errors. This evidence does not qualify native selector-5 CGLS and argues against enabling this path without native validation and an observed benefit.

## Scope limits

- The sidecar's restart-augmented consumer view is not represented or admitted here; only the current unaugmented base arrays are used.
- The source151 capture/reference does not prove which regional map FGMRES consumed at a callback. The runtime overlay applies the same-Newton base arrays directly in its explicit CGLS path, but native arbitrary-vector S/Sᵀ parity remains untested.
- Existing CGLS binary state-capture v1 is rejected when `regional5` is requested because it does not serialize graph factors/maps. JSON direction tracing labels this map `regional5_base_graph`.
- The source464 CPU result is formula-only analysis with a reference clone's completed-tail latch cleared; it is not native CGLS, solver convergence, physical qualification, or a performance result.
