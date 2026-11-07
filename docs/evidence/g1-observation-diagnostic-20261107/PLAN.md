# Delayed-action observation investigation

In the historical G1 legs task, the actor frame names 12 delayed-action
operators with delay component 1, followed by command, joint position error,
finite-difference joint velocity, gravity, gait phase and local angular
velocity. There are five 48-value history frames, oldest first.

The retained 4081010 policy rollout contains actor observations and extracted
raw actions. For environment 0's first episode, the last actor frame's ankle
position error exactly matches the preceding trace's ankle position, and its
projected gravity agrees with the preceding trace's quaternion within
8.65e-8 over 372 samples. From pre-step 3 through 373, all 371 expected
delay-one left-ankle raw actions are nonzero, while the 371 corresponding
observations are zero. This comparison uses the owner code's history layout:
at completed step k the raw-last slot holds action k and the preceding slot
holds action k-1; the resulting actor frame is read at step k+1. A raw command
is not a filtered actuator value or joint target.

There is a separate unexplained discrepancy at actor sample 65: the published
left-ankle finite-difference velocity is 4.06873274 rad/s while the adjacent
emitted position samples imply 0.433714315 rad/s. All other 370 comparisons
before the first terminal differ by less than 1e-5 rad/s. Environment 1 first
terminated at step 63; its subsequent reset is a temporal coincidence, not
evidence of causation. The first episode of environment 0 itself terminates at
step 373. The actor observations and recorded poses have different intended
timing; the gravity and position checks establish the described alignment,
but do not independently validate the complete reset or velocity contracts.

The historical and current source uses the following expression for
`delayedAction`:

`previousAction[operation.source.y - operation.source.z * program.counts0.x]`

The operands are unsigned 32-bit fields. The CPU oracle shows that for 29
actions, raw slot 1, action index 5 and delay 1, the intended relative element
is 5, whereas the subscript wraps to 4,294,967,272 elements. The wrapped offset
is not a negative pointer displacement. This source analysis and the observed
zeros are consistent with an invalid access, but do not themselves prove a
particular Metal device's memory-access behavior. They do not establish why a
policy falls. Reset-frame evaluation uses a different history pointer, so a
fix must also define reset history instead of blindly replacing the subtraction.

## Bounded native fault-isolation plan

The preregistered owner-level probe supplies known nonzero Float32 actions to
the exact historical G1 task, then reads actor observations from the task's C
API. It verifies the same four semantic fingerprints and uses the retained
historical native executable library and metallib. It runs eight 20 ms control
steps, eight environments, four physics substeps and 4+2 contact iterations.
There are no learned-policy or perturbation claims.

At steps 3–8, the 12 leg delayed-action observations in each environment
should equal the previous-previous supplied commands within 1e-6, unless that
environment was reset. On step 5, environment 1 receives a reset request;
its observations from step 5 onward are excluded from this particular value
oracle, while the seven other environments must retain their own histories.
All successful native statuses and no task termination are prerequisites.
The probe prints each compared sample and returns failure on any mismatch;
no fallback or threshold adjustment is allowed. Inputs, exact executable
and library/shader hashes, stdout, stderr, exit and the 120-second timeout will
be retained. The test is not a proof of all reset semantics or physical validity.

No GPU run has yet been made. The native result is required to isolate this
finding before deciding on a corrective implementation. A negative result is
retained and forces a reassessment of the source/timing assumptions. The
compiled executable SHA-256 is
`953e9394369db8652c975fcaeec607b4a92017fdefe4f42fc8dc9485b96ce007`.
The owner source identity and original policy/rollout evidence remain as in
the replay-gates report. Actor data is already contained in the retained
rollout pack; no new observation or policy contract is invented here.
