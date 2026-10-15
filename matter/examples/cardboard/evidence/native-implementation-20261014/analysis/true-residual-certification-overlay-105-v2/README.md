# True-residual certification candidate (source104 base)

This isolated candidate adds an opt-in `NM_FEM_TRUE_RESIDUAL_CERTIFY=1` mode. At each restarted-FGMRES cycle it defers the Arnoldi completion latch, accumulates the cycle correction, applies the same bounded explicit-FEM tangent action (including the configured tangent cache when enabled), computes a fresh `b-Ax`, and uses the unchanged forcing and absolute nonlinear thresholds. If the fresh norm misses both thresholds, it replaces the restart residual and continues within the existing linear iteration budget. The cycle-column count remains clamped to the remaining authored budget. Invalid/nonfinite true-operator work propagates to production status; unsupported scope, missing buffers, and more than 64 restart cycles reject the opt-in before dispatch.

The scope is deliberately bounded to the existing one-substep, one explicit-FEM-object path, with no MPM, mixed/vascular/Human rows, articulated generalized unknowns, adaptation, or mutation. The current operator path contains bulk/field, cohesive, rigid-contact and deformable-contact actions for the admitted row layout. The new option does not change default behavior. Host trace readback remains controlled by the separate existing root/Newton observer selectors.

The finite best correction may still proceed to the existing outer Armijo and nonlinear certificate when the authored restart budget is exhausted without true linear convergence; `diagnostics.z` remains false, and no tolerance or physical acceptance gate is relaxed. Therefore this candidate can expose/avoid false Arnoldi completion but does not guarantee convergence or cure FP32 conditioning. The capture360/source101 direct-solve diagnostic found native correction relative error 0.008729 and FP32-cast true residual 1.29786e-5 Ns versus 1.19929e-5 Ns forcing target; retain unresolved finite-floor outcomes.

The new native kernel fixture dispatches the production finalize, correction-accumulation, and true-residual kernels. Its small algebraic action is `A=I`, with the operator result supplied from the correction actually accumulated by the production kernels. It checks deferred latch, residual replacement, successful true-residual completion, nonfinite propagation without partial replacement, and preservation of an existing production failure. This isolates the kernel state machine; it is not a full FEM operator or physical qualification.

## Validation

- `python3 matter/tools/test_true_residual_certification_contract.py`: 10 CPU source/contract tests passed.
- `ctest --test-dir build-v2-fresh -R '^matter.analysis.fgmres_true_residual_contract$' --output-on-failure`: passed.
- `cmake -S source -B build-v2-fresh -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`: configured.
- `cmake --build build-v2-fresh --target numi_matter_shaders numi_matter_runtime_objects numi-matter-fgmres-true-residual-native-check -j2`: passed; Metal emitted two existing initialization warnings in graph-Schwarz code (`sharedValid`, `refinementValid`), unchanged by this patch. The native fixture was compiled only, not executed.
- No GPU/native solver run was performed; this is not native numerical or physical qualification.

Patch is `source-overlay.patch`, relative to frozen source104 at `snapshots/source-contact-health-diagnostics-104`. `manifest.json` binds every changed/new source file and validation artifact.
