# Longer fresh G1 training comparison

Registered before trials. The 20-update pilot failed its preregistered skill
screen. Test whether a larger but still bounded training budget changes the
outcome on the same installed, corrected-observation G1 task, without importing
historical incompatible policies or changing physical settings.

Run three independently initialized policies using physics seeds 20261201,
20261202 and 20261203 and learner seeds 20261211, 20261212 and 20261213.
Each uses 16 environments, 24 steps per update, 200 updates, chunk 8, two
update epochs, ground scene and the exact staged 3a439b63 package. Use a
matched incumbent/candidate comparison with eight environments, full 1000-step
episode horizon and selection seed 4081021. Training, selection, task and
failure records must all complete. Each trial has its own run directory and
180-second process-group timeout; preserve every failure or partial result.
Other task GPU work is paused for each run. CPU and system activity are
recorded but not controlled.

Prediction: increasing the optimization budget will produce a candidate with
a positive matched selection score in at least two of three runs and no
device-failed environment steps. This is an exploratory prediction, not an
assumption that training will succeed.

Independently apply the same initial skill screen from the pilot: no failed
physics steps, all eight environments with a clean horizon, and at least one
metre final progress in each environment. Selection seed 4081021 is used during
candidate selection and cannot serve as an untouched final evaluation. A pass
requires separate new evaluation seeds and disturbances, sensor and actuator
qualification, and multiple training seeds before any general claim. A
selector-advanced policy with no clean horizon is not physically qualified.
Do not equate successful device execution, positive mean progress, or selector
advancement with successful locomotion. Capture full hashes, outputs, policies,
optimizer state, host snapshots and exact wall time.
