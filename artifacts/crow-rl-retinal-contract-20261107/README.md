# Retinal actor contract and native admission

This checkpoint defines the native v13 retinal feeding contract and protects
its executable boundary. It does not yet deliver rendered images to the actor.
A v13 rollout/training CLI is not exposed. The next required implementation is
the real two-camera GPU observer, native callback composition, visibility and
delay controls, a pixel-dependent actor, and complete performance/learning tests.
The complete lifelong environment remains unqualified.

The distinct contract has 1,251 actor features: 97 existing channels, two
16 x 12 RGB planes, valid and age. The critic remains 97 and actions remain 18.
Mechanics, action meanings and finite-resource program identity are unchanged.
Camera positions/orientations, source/output dimensions, optics, sampling,
latency, content and response semantics bind both observation and task
fingerprints. The source is SensorPack, not an objective or hidden teacher.
New native observation metadata requires the exact observation fingerprint.

Shared task structs and legacy compilation retain ABI/tag 47. Retinal tasks
receive executable tag 48, supported explicitly by task kernels. The host
checks a capability in the same task-shader translation unit before submitting
48. The old ABI-47 shader is retained, with its exact receipt, in inputs/ for
portable negative controls. It is an actual prior native shader, not a mock.
C API structs are unchanged; C++ consumers need a rebuild for the new observer
metadata. No legacy policy is dimension-rebound or silently promoted to v13.

Trial 01 passed compiler, sensor-contract and native missing/stale checks.
Trial 02 added exact observer identity and a mismatch control, and removed two
instrument compiler warnings. Its 12 malformed manifests reject, and nine
camera/content/timing changes with the same 1,251 inputs reject the old policy.
Asset path relocation with identical content preserves identity. Native missing
and mismatched observers and the old shader are rejected before a live physics
submission. The current shader reaches the expected callback layout, whose
instrument deliberately returns false rather than pretending to supply images.
The fixture visual paths/hashes are compiler metadata only; no visual files are
loaded and no actor images are rendered by this probe. These are instrument
controls, not animal trajectories, learning or full environment qualification.

All nine focused native CTests and 19 Python crow checks passed. CTest output
and native evidence directories are copied here. The frozen v12 feeding actor
retains both exact prior replay fingerprints, ordinary and residual, under
chunks 1/8 and the original initial deterministic base. Thus legacy task,
observation, action and accepted rollout behavior remain unchanged after the
additive semantic tag and shader capability change.

Trials retain exact source deltas and untracked source against commit 461d7b73.
Archive creation rechecked their source and stdout/stderr hashes. The exact
current executable/library/shader trio remains on macmini under the paths in
runtime-receipt.json. Source, old shader input, controls, retained CTest images
and state and raw reports are in Git. payload-inventory.json binds copied data.
