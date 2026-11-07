# American crow: native head-camera geometry

Two estimated parallel head cameras now follow accepted active-head poses.
This is publication-time geometric calibration; the v12 learner still has
97 non-image inputs. It does not establish biological eyes, sensor cadence,
latency, learned vision, search, or an ecological environment.

Attempt 04 passed the prospectively extended instrument: 198,286 stable
geometry pixels and 92,374 stable empty pixels, zero identity/validity errors,
maximum optical-z depth error 55.83 micrometres (limit 200), and maximum camera
pose component error 8.51e-8 (limit 2e-6). Neutral views have 360 target pixels
per camera/lane; physical occlusion and averted yaw hide it. Physical pitch
agrees with the independent FP64 geometry oracle. Accepted yaw body states
are bit-identical under chunks 1/8. Every RGB/depth/identity/validity render
repeats exactly. Seven malformed-authoring controls reject invalid input.
Two lanes, four controls and one partition repeat are calibration units,
not independent animals or learned behavioral evidence.

Attempts 01 and 02 failed before rendering: missing tangent sign and wrong
primitive-to-instance ownership. Their failures remain unchanged. Attempt 03
passed the original geometric prediction. Attempt 04 added empty-space and
invalid-authoring controls with unchanged original tolerances. Raw images,
poses, depth, identities, validity, visual assets and PNGs are archived.
Per-camera layout JSON accompanies attempt 04 binary records; native headers
own structured body layouts. PNGs are sRGB display previews of linear RGB.
Collision meshes exactly retain boxes and bounded +Y planes (16 m half-extent)
and do not change the physical world. Camera positions, parallel directions,
45 degree FOV and optics are engineering estimates, not crow measurements.

All seven focused native CTests and 19 Python crow checks passed. The frozen
feeding actor has unchanged ordinary and residual replay fingerprints when
using the original initial deterministic base. Feeding replay01 used the
trained deterministic base instead: it passed partition invariance but is
not the prior residual-reference comparison. Replay02 uses the original base
and matches both prior fingerprints exactly. Both results remain retained.

Replay02 includes a complete source delta against commit 543e59ab, including
new native files, with hashes verified again during archival. Earlier trials
pre-date automatic source-delta retention. verified-authoring-inputs/ contains
only reconstructed older files whose hashes exactly match their original
plans; source-recovery-receipt.json states which were recovered. The complete
attempt01 app source is not reconstructed. Its failed schema validation is
retained evidence, not a reproducible successful scientific result. Current
source and the complete passing delta are retained. Runtime-receipt.json
identifies the exact passing binary trio retained on macmini outside Git.

The next instrument must deliver causal device-resident RGB without identity
or object-position privileges, explicit sampling/latency and complete reset/
restore semantics before connecting a distinct actor contract or learning.
