# G1 recorded-policy extraction and replay result

The preregistered gates passed on Apple M6 for fresh evaluation seed 4081010.
The closed-loop policy, policy with rollout recording, and unmodified
action-stream replay each completed 1,000 control steps in eight environments
without a reported physics failure or process timeout. The compiled policy
program fingerprint was 17940125103817627925, and the world, task, observation
and action fingerprints matched the policy contract exactly.

Recording preserved all eight first terminal steps and tilts, and environment
0's emitted configuration trace through its first terminal step: 373 trace
rows, plus the header, matched. The extracted unmodified action stream replay
also matched all 1,000 emitted environment 0 trace rows and the header, along
with the eight first terminal steps and tilts. These traces contain text
formatted to nine significant digits; this is exact equality of the recorded
text, not a demonstrated bitwise match of internal state. Other environments
were not individually traced, so their full state histories are not qualified.

All eight first episodes terminated before step 1,000: steps
[373, 63, 905, 678, 404, 517, 368, 250]. This is an instrument-validation
result, not evidence of successful locomotion or physical deployment. The
full set of commands, outcomes, traces, the 12 MB rollout pack and extracted
actions, source patch and SHA-256 identities are retained in this directory.
The original historical policy pack, compiled binary and reconstructed owner
remain host-only dependencies.

## The preregistered intervention did not run

The sealed intervention plan and generator were committed before this seed was
observed. Its trigger ended at step 342, giving a proposed window of steps
343–352. The generator rejected both arms under its predeclared [-1, 1] action
admissibility rule and produced no intervention streams. At the first step the
baseline left-ankle action was -1.1494827; adding or subtracting 0.1 would
remain outside the range. Six of the ten steps would violate the rule for each
sign. The preparatory command and nonzero exit are retained. There is no
intervention outcome and no causal benefit can be inferred.

The historical task explicitly permits policy actions outside [-1, 1]; it
passes them through action history, possible delay and response filtering, then
clamps the physical joint-position target to its configured limits. Thus the
sealed input bound was conservative, and failure does not mean the original
policy produced an illegal physical target. It also does not establish whether
a replacement pulse would reach an unsaturated actuator: the emitted trace does
not contain the selected action history, filtered action, unclamped/clamped
target, drive effort, or contact forces.

A defensible next intervention needs an owner-level measurement of those
per-environment values, their timing and joint-limit margin, with the angular
velocity and foot contact response needed to interpret the tilt. Qualify the
measurement against accepted state and unmodified replay before designing a new
pulse on unused evaluation seeds. The present failed design is preserved and
must not be retuned after inspecting this seed.
