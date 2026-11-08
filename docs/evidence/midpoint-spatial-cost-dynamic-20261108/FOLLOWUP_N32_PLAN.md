# Second independent spatial follow-up, preregistered before execution

The fixed-law 8-to-16 mesh pair failed its original spatial gates while both
references passed the 0.5/0.25 us temporal check. Test one 32-division mesh
(35,937 nodes) with exactly the same geometry, material, gap, activation
width, reference-area contact coefficient and duration. Keep the 2 us trace
clock and independently check temporal convergence at 0.5 versus 0.25 us.
Predict the 16-to-32 pair will still fail at least one of the original four
spatial gates; reject that prediction if all pass after temporal qualification.
Do not change thresholds after observing the result. The existing FP64
reference's input limit is raised only to permit this mesh; compile it as a
separate CPU executable, run its self-test, and verify that its 2-division
output matches the prior 0.25 us reference before using it. Record source,
binary hashes, completion and elapsed time. Run with low CPU priority.

This is a CPU reference, not native n=32 Matter qualification. Stop at n=32
for this follow-up, retain failed or incomplete runs, and avoid running native
GPU cost benchmarks concurrently with the CPU reference. A new refinement
requires a separate resource and validation plan.
