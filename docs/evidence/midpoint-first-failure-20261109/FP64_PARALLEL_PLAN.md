# Bounded acceleration of the independent reference

The preregistered n=64 reference is running at low priority on one CPU core.
To shorten the second temporal run without changing the mathematical model,
prepare a separate reference executable that evaluates element contributions
in parallel and gathers nodal forces in precisely the original tetrahedron and
within-tetrahedron order. Gravity, contact pressure, impulse accumulation,
RK4 operations, output sampling and tolerances remain unchanged. Limit it to
four low-priority worker threads in one process. Do not run it concurrently
with GPU timing benchmarks.

Before using it for any reference, its energy-gradient self-test must pass and
its complete n=2 and n=8 0.25 us outputs must match the retained single-thread
reference byte for byte. Then its complete n=32 0.5 us output must match the
retained output byte for byte. Record executable/source hashes and elapsed
wall time. Any discrepancy, failure or lack of speedup disqualifies it; retain
the attempts and continue with the original single-thread executable. If all
checks pass and it is faster, this separately preregistered plan amends the
n=64 plan only for the second 0.25 us run; the active 0.5 us run stays on the
original executable. Validate the resulting n=64 pair using the original
temporal and spatial gates. Matching smaller cases supports but does not prove
identity at n=64, so the different execution path remains explicit in reporting.
