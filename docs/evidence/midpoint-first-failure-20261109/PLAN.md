# Midpoint precontact failure: preregistered diagnostic

The retained 8-division, 3.90625 us midpoint run rejected at step 203,
before contact, with the physical momentum certificate narrowly over its
limit. Do not alter the model, force law, solver tolerance, integrator or
acceptance checks. Reproduce the 203 accepted steps and capture the next
step's accepted state, final midpoint candidate, reflected endpoint and
assembled GPU residual using a borrowed command buffer. Capture is explicit
and diagnostic; no scratch buffer becomes accepted state. Verify the same
failure category and exact restoration.

In independent FP64 arithmetic evaluate (a) twice the stored stage residual,
(b) the endpoint velocity-reflection correction, and (c) their sum. Compare
that sum to the device certificate and independently recompute the complete
midpoint force and momentum residual from geometry, material and gravity.
Prediction: reflected velocity rounding contributes materially to the
failure. A dominant stage residual or disagreement with the independent
physical residual rejects that explanation. Record individual norms and
maximum nodal errors; no single aggregate cancellation is sufficient.
Retain the capture and the result even if the prediction fails. The new
buffer views are available only for the already opt-in midpoint diagnostic
fixture and are read-only, borrowed, and limited to the command-buffer
lifetime. They do not loosen the fixed-plane or coupled-owner guards.
