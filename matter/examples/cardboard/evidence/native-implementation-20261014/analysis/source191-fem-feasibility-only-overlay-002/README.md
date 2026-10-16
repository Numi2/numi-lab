# Source 191-002: fail-closed FEM-only deformable-contact feasibility gate

This isolated follow-on preserves source191-001 and is based on the immutable source 189 buildfix003 snapshot. It tightens the policy classification to require the exact FEM-only support mask; unknown representation bits fail closed. It does not edit the checkout or launch a GPU.

The new branch is off by default because its host flag is one only after NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT is selected and its existing full-contact scope checks succeed. With that flag, only a deformable contact whose validated support contains FEM nodes and no MPM nodes can skip the per-feature energy-decrease inequality. It still requires a finite current barrier evaluation, finite trial barrier evaluation, and the existing full CCD certificate for each exact femTrialNodeAtAlpha endpoint. MPM-only and mixed FEM/MPM supports keep the existing feature Armijo path.

The global full-contact loop independently rebuilds private deformable contacts and invokes nm_contact_validate_armijo_deformable_sweep for each exact trial candidate, then measures the coupled free-DOF residual and makes the existing Armijo decision. The applied solution uses femTrialNodeAtAlpha with the same selected alpha, and the ordinary final full-state residual/certificate path remains unchanged. No tolerance, force, tangent, contact energy, CCD threshold, or physical acceptance criterion changes.

## CPU checks

policy_contract.cpp calls the shared inline helper that the Metal kernel itself calls. Seven cases verify default-off, exact FEM-only, MPM-only, mixed support, no support, disabled full-residual mode, and unknown support bits. A static source-contract audit checked the host binding, exact endpoint/full-CCD branch, global trial sweep, residual measure, and final certificate remains present. This is a candidate package only; no Metal compilation or GPU dispatch was performed.

This does not demonstrate that the policy cures the loaded-contact failure, nor does the 511 helper constitute a physical Matter transaction. Native end-to-end validation is still required.
