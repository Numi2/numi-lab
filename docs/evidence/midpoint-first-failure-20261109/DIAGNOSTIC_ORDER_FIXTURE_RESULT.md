# Fixed-plane diagnostic-order and retry check

The native midpoint probe now injects the same nonlinear failure category
with two distinct offending-element indices and associated diagnostics after
the candidate is encoded and before post-commit reconciliation. Each attempt
starts from the same accepted snapshot. It checks the retained category and
index, all compared physical state and transaction cursor after rollback, and
the physical state and failure category after the subsequent real retry,
against an uninterrupted continuation. The index is not canonicalized.

On the native 27-node fixture, both the initially active 95 micrometre gap and
the initially inactive 200 micrometre gap passed. The first accepted 95
micrometre step had nine contact samples and a 3.61695950346e-6 N s normal
impulse; the 200 micrometre step had no contacts. See
diagnostic-order-injection-001/002 execution records and complete traces for
source, binary, shader, inputs and outputs. The injected diagnostics exercise
two specific choices; they do not prove all failure sources equivalent.

This test covers the fixed-plane Matter transaction. It does not qualify
spatial convergence, long-rollout force accuracy, or the still-disabled
reacting-body transaction. The long-running backward Euler control uses the
separately frozen earlier binary and is not relabeled by this test.
