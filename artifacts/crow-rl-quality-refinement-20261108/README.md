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

## Observed qualification

Full Release succeeds. All25 retained-problem submissions pass their declared
comparisons: the three newer retained failures now satisfy original tolerances
and independent audits; substep2 remains successful, strict substep7 rejects
with untouched outputs, old smaller arenas reproduce baseline bytes, and
repeats are exact. The native unified/IR/contact owners pass. All624 motor
physical records and ten default-crow physical files remain exact. The strict
physical capture differs only in its declared expanded Hessian stride.

The coupled API-default crow now completes control tick0 in all four lanes.
During tick1, one lane completes and three reject: first failure is environment0,
substep8, quality status6 (line-search failure), after17 Newton/239 PCG iterations
and66 backtracks. The current refinement handles only status7, exhausted Newton
budget; it has not handled this line-search failure. Both trajectories match
exactly. The first four sampled states have maximum jaw penetration2.18243e-7m,
but these four samples do not bound the whole physical trajectory or the partially
accepted next tick. The640-transition completion prediction is contradicted.
No profile, policy or full environment qualification is promoted.

The inspector cannot supply a matching quality witness for this new stop:
the retained stderr explicitly reports protected-witness/public-failure mismatch.
Existing capture admission covers status7, not this first status6 failure.
Preserve these results before extending capture coverage and independently
checking the failed operator. Do not infer a precision cause from the aggregate
line-search status or simply extend solver budgets.

The source candidate is retained because old qualified behavior remains exact
and new publications meet the same three certificate gates. Integration,
new-failure handling, reset/checkpoint partition behavior, biological fidelity
and end-to-end learning performance remain unfinished.
