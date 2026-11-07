# Fresh G1 rigid-learning pilot

Registered before execution. Question: can the integrated installed trainer
complete a short fresh-policy G1 training run with the corrected delayed-action
observation contract, then compare the resulting candidate to its immutable
incumbent on matched held-out conditions? This is a feasibility pilot, not a
claim of learned locomotion or time-to-skill.

Owner is the separately staged relocated package sourced from revision
3a439b63, with the G1 delay repair and observation fingerprint guard. Record
exact package binaries, shader, Python interpreter, inputs, commands, run
directory, elapsed wall time, and errors. No historical delayed-action policy
is imported. Train on ground with g1-legs-locomotion, 16 environments, 24
steps per update, 20 updates, chunk 8, two update epochs, physics seed 20261108,
learner seed 20261109, and a newly initialized policy. The installed selector
compares candidate and incumbent with 8 environments, physical episode horizon,
and held-out seed 4081019. Fix the trainer and selector configuration before
execution; retain rejected steps, terminations and both policy outputs.

A successful pilot means the command and selector complete, no numerical
failure is hidden, and incumbent and candidate results are inspectable. That
does not mean learning succeeded. For an initial demanding skill screen, require
zero physics failures, all eight environments completing the full horizon
without height/tilt termination, and at least 1 m final forward displacement
per environment on this held-out seed. This screen is not a statistically
adequate promotion gate: a pass requires further untouched seeds, independent
training seeds, disturbances, actuation and observation qualification.

Use a 180-second process-group timeout and retain partial outputs on any timeout
or failure; a timeout is not a completed training run. Other task GPU work is
paused for the measured execution. Record host processes and exact artifact
hashes. The task has idealized simulation observations and does not qualify
deployment perception, hardware or physical fidelity. Neither experimental
contact law nor interval-local CCD is enabled.
