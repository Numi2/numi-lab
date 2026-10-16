# Independent source210 runtime and FGMRES composition review

## Scope and result

Read-only review of the frozen source210 runtime and FGMRES composition, limited to the source207 free-box true-residual behavior and source209 strict-inner-forcing behavior. No source, build, publication-checkout, or native files were changed. I found no blocking source-composition defect in the reviewed scope.

Frozen source210 is 819 files with frozen-source SHA-256 477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931. The reviewed files are matter/src/runtime.mm (SHA-256 e8a0c78dc812b491781a8f3d72a4a42d1f70e18ab3b37eff6fe157eadb7e75e1) and matter/src/metal/fgmres.metalinc (SHA-256 c1c25769c2115cd64ef9dd849cc4a987563f8d51e03638c52a1c7526b2c91c00).

## Composition checks

The source207 rigid-body residual dispatch block in runtime.mm is byte-identical to the source210 block: both the nm_fgmres_apply_free_rigid and nm_fgmres_apply_primal_rigid_contacts dispatches remain, in that order, with buffer barriers between them. The selected block is 2,390 bytes and has SHA-256 e36b85a46fffca980958eabc5ef51ec55ffa14fa5360430bafbd75d7ae25b90b.

Source210 retains source209's strict-mode admission block exactly (SHA-256 aace98527a6f50cc2cb890d447e9b69e7155c27a2dbccc273b7e62f81f0c1e17): the selector accepts only 0 or 1, and strict mode is rejected unless true-residual certification is requested. NM_FEM_STRICT_INNER_FORCING is present in the source210 solver-environment key registry.

The FGMRES completion predicate from source209 is unchanged in source210 (SHA-256 bbdf054440f5b4dc72c07adc59a4fb28e5d57d34f6ed83dc68f795381b16fd4b). The runtime supplies residual-replacement bit 0 plus strict bit 1 only when restart replacement is active and strict mode is requested (3u; otherwise 1u or 0u). For a still-active right-hand side, strict mode removes only the outer nonlinear-threshold shortcut from the inner FGMRES completion latch. It preserves exact-zero and unchanged forcing-target completion, and leaves the outer nonlinear acceptance and final certificate paths intact.

The source207 free-box true-residual scope checks remain joined to the strict path. The final source explicitly combines the free-rigid operator dispatches followed by source209's strict-mode comment and assignment. reviewed-runtime-conflict-resolution.json records this decision and says the physical operator was not modified beyond source207.

## Validation and limits

The source210 build receipt reports success; the 42 CPU contract tests and two checkpoint CTests passed. The combined source210 native helper reports both helper commands exit 0, 154 checks, and zero physical transactions. These checks cover source/dispatch behavior and synthetic helper execution; they do not establish a full coupon trajectory or physical qualification.

The saved composition check records that its automated runtime patch application and three-way check failed at the overlapping context. I treated that as a retained failed composition attempt, not as a passing test. The final manually resolved source was independently checked by exact selected-block comparisons and by inspecting the final source bytes. No full-coupon result is included in this review.
