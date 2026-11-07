# Resumed G1 training diagnostic

Registered before execution. The three fresh 200-update policies all failed
the clean-horizon screen on their shared selection seed and two additional
seeds. This tests whether a larger training budget improves the best observed
survival candidate under a matched evaluation, without treating a relative
selector win as qualification. The source is trial 3, selected because it had
the fewest height-or-tilt termination events on its selection seed. This is
one selected seed and is not a multi-seed efficacy study.

Resume its candidate PolicyPack and exact saved optimizer state for 400 more
updates in a new run directory, using 16 environments, 24 steps, chunk 8, two
update epochs, ground scene, task g1-legs-locomotion, the original physics and
learner seeds and exact installed revision 3a439b63. The simulator starts new
environments; this is policy/optimizer continuation, not a bitwise continuation
of the prior physical trajectories. Inputs and output paths are distinct and
input hashes must stay unchanged.

Use the installed matched incumbent/candidate selector at 8 environments,
1000-step horizon and previously unused-in-this-study seed 4081024. Before
considering the trial complete, verify from worker evidence that optimizer
state was restored; otherwise retain the run as invalid for this prediction.
The prediction is that the resumed candidate reduces height-or-tilt termination
events by at least 20% compared with the source actor on this matched selector
seed, with zero reported physics failures. Independently apply the earlier
screen: all eight environments must complete cleanly and each have at least
1 m final progress. A positive selector score or a successful trainer is not
that screen. If the screen passes, untouched seeds and repeated training
seeds are required before any robustness claim.

Use a 180-second process-group timeout. Retain all failures and partial
artifacts, exact hashes, elapsed time and both selector outputs. Other task GPU
work is paused during the run; CPU and system activity are uncontrolled.
The observations remain idealized and the result cannot establish physical
accuracy or real-robot deployment.
