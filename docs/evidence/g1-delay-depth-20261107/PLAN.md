# Corrected G1 delayed-action depth qualification plan

Use the corrected, separately copied historical owner and shader from
`g1-delay-fix-20261107`. This is a customized internal task fixture: it
sets maximum action delay to two steps, and selects delayed-action observation
components 0, 1, 2, and 3 in four separately compiled contexts. The fourth
component is the highest declared history slot: the compiler allows
`maximumActionDelaySteps + 2` component values. This is not the original
retained PolicyPack task and no policy is rebound.

For each depth, run eight environments for eight 20 ms control steps, using
four physics substeps and 4+2 contact iterations. Supply uniquely patterned
29-channel action vectors. Reset environment 1 on step 5. For all twelve leg
observation components in all eight environments and all eight steps, expect
the command from `step - 1 - depth`, or zero before that command exists or
if it predates environment 1's reset. Require the seven other environments
to retain their histories independently. This is 3,072 comparisons, absolute
tolerance 1e-6. All per-step status and termination checks must pass.

Run the single probe once with a 120-second timeout. Retain all output rows,
per-depth compiled world/task/observation/action fingerprints, source,
executable, linked library and shader SHA-256, command, exact UTC start/end,
and exit. A mismatch or failed execution is a failed result, with no retry or
adjustment. The probe's pre-run binary SHA-256 is
`30f1020d7f200a73210b6e5a217a21fd57185ba39bb34880c9192a9a3f5d3987`.
Native execution is pending an explicitly coordinated GPU slot.

This assesses actor-frame history and a mixed reset for the four declared
depths in this one owner fixture. It does not test actor-current, critic,
physics rejection, visual sensor history or policy performance.
