# Delayed-action repair verification

The evidence and failed native historical observation probe are retained at
commit d8c2a477. The current-branch repair indexes delayed actions from the
environment-owned history base and separately retains the existing
previous-action pointer. Reset clears every history slot; actor and critic
frame/current evaluation receive the same owner base. The observation
fingerprint changes only when actor or critic operators include delayed
actions, so old PolicyPacks must not silently acquire different observations.

The current checkout is sparse and its CMake configuration fails on unrelated
missing sources. The equivalent two-part repair was ported to a separate copy
of the complete historical owner; both port diffs are retained here. Its
native library and shader compiled. The current branch's changed Metal shader
and TaskProgram.cpp also passed standalone compilation/syntax checks.

CPU contract check on the corrected historical owner: world, task and action
fingerprints match the old PolicyPack, while observation fingerprint changes
from 15900562835394482329 to 9355055053806751635. The compiler rejects the
old policy as incompatible. A separate no-delayed-action G1 velocity task
compiled identically against old and corrected libraries for all four
fingerprints. This is not a new policy qualification.

## Preregistered bounded GPU verification

One eight-step, eight-environment owner-level run with known 29-action
sequences, same 20 ms timing, four substeps and 4+2 contact iterations as the
baseline. Use `native_action_repaired_probe.cpp` against the corrected
historical library and shader; require world, task and action fingerprints
unchanged and corrected observation fingerprint 9355055053806751635.

Compare all 12 leg delayed-action observations in all eight environments
over all eight steps, 768 comparisons at absolute tolerance 1e-6. For each
environment, steps 1 and 2 must be zero after initial history fill; steps 3–8
must be the action supplied two steps earlier. On step 5, reset environment 1:
its observations at steps 5 and 6 must be zero and its step 7 and 8 values
must be the post-reset actions from steps 5 and 6. Other environments retain
their independently expected history throughout. Require no physics failure
or task termination. Retain every row, source, command, binary/library/metallib
hashes, exit and 120-second timeout. No retry or policy rebinding.

Passing establishes this historical owner's observed delay-one path for this
bounded mixed-reset fixture. The CPU oracle checks index arithmetic at zero,
one, and maximum supported delays but does not qualify Metal for other delay
values, commands, paths, policy behavior, or sensor rollback. The current
branch's full native build is not established by the historical port.

## Uncoordinated test disclosure

A developer mistakenly invoked the corrected historical
`metalrobo_task_program_check` believing it was CPU-only; the typed-actuator
fixture used Metal for ten 64-step runs. Its output was retained. The output
file birth/modification times bound the process approximately to 14:46:34–37
PST on 7 November 2026; exact process start/end timestamps were not captured.
This was not the preregistered delay verification, did not test delayed-action
history, and must not be used for performance claims.

The collision owner reported moving-contact output at 14:46:18–20 and
resident rollback output at 14:46:42 PST; the latter's exact start was not
recorded. The product owner reported asset failures at 14:46:50 and corrected
native probes at 14:47:02; its earlier baseline lacks exact timestamps, so
overlap with that baseline cannot be ruled out. Neither owner makes performance
claims from those runs. Both owners were notified immediately.
