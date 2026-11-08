# Checked native quality refinement: qualification pending

Small otherwise nonconverged quality problems can use a bounded two-FP32
refinement through the existing shared solver wrapper. The candidate rounds
velocity to FP32, reprojects accurately, requires the existing FP32 evaluator,
and accurately audits the actual published impulse stream. All three gates
must meet unchanged dispatch tolerances. Old smaller arenas and primary
budgets below20 skip refinement; already converged solves keep their path.
ABI1 record sizes and canonical FP32 publication remain unchanged. SolvePath2
and total Newton/backtrack counters expose refinement work.

The parallel triangular dependency fronts replace the research instrument's
lane-zero triangular solve. An expanded Hessian arena supplies all device
scratch, leaving the existing threadgroup allocation unchanged. This must pass
retained-problem parity, old-arena rejection, owning native regressions,
full physical crow trajectories and end-to-end performance qualification.
No protected policy or RunProfile is promoted by this source change.
