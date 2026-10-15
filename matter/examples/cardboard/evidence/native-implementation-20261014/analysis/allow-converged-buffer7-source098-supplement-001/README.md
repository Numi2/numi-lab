# Explicit buffer-7 binding supplement

This is an additive supplement for `center-residual-jdelta-source098-candidate-001`; it does not edit that candidate or frozen source098.

`nm_fgmres_apply_primal_rigid_contacts` and `nm_fgmres_apply_free_rigid` both declare `allowConvergedState` at Metal buffer 7. Direct calls in the angular and coupled free-box native checkers previously stopped after buffer 6. This patch binds a zero-valued `uint` at buffer 7 for each call. Zero preserves normal production behavior (a converged FGMRES state is not allowed through the operator). No shader, solver, tolerance, or CMake changes are included.

A source-wide search found exactly two checker files that directly dispatch these kernels and four dispatch sites total. `verify_buffer_bindings.py` checks each site. `verify_candidate_corrected_copy.py` validates the unchanged integration candidate. Its temporary verifier copy also corrects two verifier-only schema-key mismatches: the candidate records `new_array_count_max` and `payload_cap_bytes`. The original candidate verifier and package remain untouched.

Validation is static only: strict Objective-C++ syntax checks for both checkers, candidate structural verification, binding regression, and patch dry-run all pass. No GPU execution was performed. The separate rich-RAP per-environment stride correction and root-owned probe writer are not included.
