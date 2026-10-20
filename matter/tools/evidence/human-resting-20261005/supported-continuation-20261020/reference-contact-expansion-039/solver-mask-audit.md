# Source audit: FEM preconditioner selector and exact VT JVP masking

This is a source-level audit for the 039 terminal replay. It does not infer an active fixed/nondue-row defect from the terminal contact trace.

The Human app creates `WorldSource` and sets Newton, FGMRES restart/iteration, and line-search budgets at `apps/numilab_human_myosim_visual_probe.mm:21772-21779` and `:23029-23036`. It does not assign `mixedSolver.femPreconditioner`. The default in `matter/include/numi/matter/matter.hpp:376-420` is `FEMPreconditionerMode::scalarDiagonal` (selector 0); selector 6 is the separate coupled FEM tangent block Jacobi mode. `matter/src/compiler_impl.cpp:281-286` copies that enum into `executionBudgets.z`.

The replay invocation sets `NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=1`. This environment variable only sets `vascularBlockPreconditionerEnabled` in `matter/src/runtime.mm:4031-4040`; it does not change `executionBudgets.z` or select FEM preconditioner 6. The source diagnostic at `runtime.mm:20973-20974` would print the FEM selector when emitted. The replay stdout did not provide that field, so the run's selector is inferred from the Human app's unchanged default construction path, not claimed from a captured runtime selector value.

The exact VT JVP helper `deformableExactVertexTriangleDirectionalForce` in `matter/src/metal/fgmres.metalinc:782-849` reads all four entries from its direction buffer directly. Production calls it from `nm_fgmres_gather_nodes` around `:5473`; the runtime binds the accumulated preconditioned direction. Fixed FEM rows are assigned zero inverse response in the scalar diagonal preconditioner at `fgmres.metalinc:3107-3119`, so the standard path should produce zero direction for fixed rows. The gather kernel clears its output and early-returns for nondue objects at `fgmres.metalinc:5387-5403`.

A conditional concern remains for cross-rate contacts: `nm_contact_accumulate_deformable_fem_residual` at `matter/src/metal/contact.metalinc:10590-10637` can add contact impulse to a dynamic node without an `objectDue` gate, while the FGMRES gather later omits nondue rows. If such a nondue node is present in an exact VT contact support, its direction entry is not explicitly zeroed by the JVP helper. The current terminal trace does not include per-support due masks, so this remains an unproven audit caveat and is not assigned as the cause of the Armijo failure.

The terminal contact is EDGE_EDGE, so the exact VT JVP helper is not the operator path selected for the failure. No source change follows from this audit.
