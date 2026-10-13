# Construction-only resting posture

`human resting-run --initial-posture COMPLETED_NATIVE_LOG
--release-initialization ...` initializes a fresh resting simulation at the
retained terminal posture. It replaces the root-pose argument and hashes the log
as an owner input. The file is bounded to 256 MiB before hashing. This option
requires the matching Numi Lab native initial-posture reader.

The native reader requires exactly one passing rigid source record with the
current rigid-payload SHA-256, one unassisted legacy terminal state, and one
completed unassisted resting footer in that order. It checks the exact generalized
coordinate dimension, finite Float32 round trips including signed zero, unit root
quaternion, positive step count/timestep, and rejects duplicate decoded JSON
member names. The current source/terminal/footer contract is used without an
alternate coordinate model or manufactured completion record.

Released full-skin GPU contact is required. Other root/joint overrides and terminal
inspection cannot be combined with this initializer. Ground alignment is skipped
for this explicit world posture, and native recruitment must preserve every
coordinate exactly. Source initial velocities must be zero. Muscle recruitment,
respiration, circulation and controller state are initialized afresh. Prior
velocities, activations, solver history and physiology are not imported: this is a
construction posture, not a checkpoint restart or proof of prior physiology.

The launcher admission tests cover release/missing/oversized inputs, exact owner
hashes and root-argument replacement. Native admission controls and a real retained
completed 6,000-step log cover the reader independently. Real scene execution is
recorded separately and retains its source-bound qualification limits.

Native M4 Pro validation preserved all 129 coordinates of an exact retained
64-step source posture and advanced four fresh unassisted 2 ms steps with fixed
bed contact. Its nonzero prior velocity was discarded. A retained 10,000-step
posture exceeding joint 114's source upper bound was rejected before common
initialization; the source limit and 1e-7 recruitment tolerance were unchanged.
Fresh recruitment uses no legacy standing ground-plane load for this bed-owned
posture. This verifies construction and bounded advancement, not sustained
resting stability, checkpoint restart, or physiological completion.
