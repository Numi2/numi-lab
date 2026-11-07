# Live native retinal actor evidence

The native observer now composes both physical head cameras, GPU retinal
cadence/delay and policy inference in the owning command buffer. No host
pixel readback, task-clock decision, commit or wait occurs in that callback.
The controlled actor reads only one declared RGB cell for its yaw rule.

Final trial12: all six bounded controls pass, 384 accepted transitions, zero
sensor errors/physical terminations, two cold-start blind ticks and lane1
reset at tick17 with uninterrupted lane0 unchanged. Observations, actions,
latents, transitions, full q/v trajectories and final q/v/scene states are
bit-identical across chunk1/8. Physical occlusion reverses the tick2 action
from +0.426903 to -0.356149. Disconnected pixel coefficients give zero actions.
The independent FP64 delayed RGB oracle has maximum error9.72599e-8 against
registered3e-6. Eight malformed observer inputs reject. Ten native regression
checks, nineteen Python checks and the final-capacity live recheck pass.
Legacy frozen feeding fingerprints remain2255877855914022940 and
3082487550463979285. These are authored mechanism controls, not visual learning.

The fixed development observer reconstitutes camera/world/visual content from
actual physics and bound source hashes. Its estimated instantaneous parallel
pinhole cameras are not measured American-crow eyes. It currently renders each
control tick and samples on GPU at15Hz with40ms delay. Production batch scaling,
v13 C API/Swift/MLX learning, whole world/RNG/sensory/memory restore, robust
locomotion, physical food handling, physiology, memory and social agents remain
open. The lifelong environment is not qualified.

All12 attempts, exact source deltas/untracked sources, prospective plans,
receipts, raw measurements and authored actors are preserved. Attempts01–03
failed setup/admission;04–05 passed initial controls;06–07 were explicitly
stopped after repeated Metal IO loading exhausted64 dispatch workers, with
samples and exit=-15 receipts. Per-run autorelease cleanup alone did not solve
it. Shared bounded per-device loading resolves the repeat-construction stall:
08 passes all six runs in4.10s including construction and inspection.09 adds
negative/isolation/trajectory checks.10 records capacity usage.11 fails an
undersized capacity refinement;12 uses measured substep requirements (230 raw,
140 blocks), rounding to256 raw/160 blocks and matching row/endpoint bounds,
while retaining original pair/manifold/convex/CCD bounds. No behavioral or
numerical threshold was relaxed. Final-substep usage alone was insufficient.

Large raw arrays are losslessly gzip-compressed. payload-inventory.json records
both original and archived hashes/bytes. Decompress each gzip to its source
basename for inspection. RGB is environment,y,x,RGBAfloat32; actor is
step,environment,1251float32; actions/latents are step,environment,18float32.
Actors: first97 existing channels, left then right12x16x3 RGB, valid, age sec.
Body and transition record sizes are in trial evidence; q/v trajectories use
actual model dimensions23/22. No snapshot is a full-world checkpoint.

The exact final executable/library/core and retinal shaders are hash-verified
in .numi/retained-runtime/crow-retinal-live-admission-12 (outside Git due size),
with runtime-receipt.json. To reproduce, preserve its bin/lib/shaders layout
and use DYLD_LIBRARY_PATH pointing at its lib directory when running the saved
binary with a fresh output directory. Older attempts preserve exact source
and input hashes, not every historical binary. The broad native regression
log predates the capacity-only app refinement; native-live-final.log verifies
the final code. The unchanged library supports both runs.
