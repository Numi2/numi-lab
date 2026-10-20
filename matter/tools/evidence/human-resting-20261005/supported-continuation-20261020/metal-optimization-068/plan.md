# GPU-gated FGMRES tails 068

Question: does skipping the transport field polynomial smoother when a compiled world has zero `NM_OBJECT_MULTIPHYSICS` objects and zero field-boundary rows improve the exact resting-Human mechanics replay?

The control is the retained accepted four-root replay from screen 064, source `aadcd68e0747540978edd88a23af0cd114932942`, binary SHA-256 `c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828`. The treatment uses the same frozen invocation/configuration, exact scene assets, device, posture, timestep, physical materials, contacts, masses, and acceptance rules, with only the source change in `matter/src/runtime.mm`: omit the three per-pass matrix-free field smoother dispatches when no transport unknowns or field boundaries exist. Field scratch initialization/export, mechanical and vascular preconditioning, nonlinear/Newton/FGMRES limits, true-residual and rollback remain unchanged.

Prediction: the Human will report zero multiphysics objects and field-boundary rows, skip exactly three dispatches per configured field-smoother pass per Krylov column, pass all four native roots and unchanged terminal physical metrics, and run faster. This remains a short mechanics-prefix performance result, not sustained or physiology qualification.


This run also validates the GPU-authored indirect node and element dispatch extents from the FGMRES begin/finish kernels. Those extents are used only for one-environment node/element domains inside the Arnoldi column loop; finalization, true-residual validation, accepted-state publication, and rollback retain direct dispatch.
