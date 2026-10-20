# Supported mechanics continuation
Exploratory numerical development, not physiological qualification.

Question: which existing GPU stage prevents extending the supported skin/fat
and ligament scene beyond 0.5 ms, and is the initial support impulse physical
initialization error or persistent instability?

Freeze current source a143f2a0 and native009 input. First run two 125 us steps
with read-only existing Metal timestamp sampling at root 1, all kernels sampled
every 32 dispatches. Compare first two accepted physical trace rows with native009;
timing metadata must not change accepted dynamics. Use sampled stage times and
dispatch counts as a locator, not a complete unbiased profile. Unavailable or
dropped counters remain explicit. Stop on first physical failure. No visual checks.

Bound work: one diagnostic profile, one cause-directed implementation, focused
regression, then at most two continuation runs (8 and 32 steps) if admission passes.
Retain failures; do not relax tolerances or replace physics. New ligament source
work proceeds independently, but no competing GPU jobs. No long qualification
run until these bounded checks identify a viable supported state.
