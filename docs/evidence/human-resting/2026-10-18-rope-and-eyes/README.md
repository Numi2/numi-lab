# Rope suspension and visible source eyes

The existing native Human scene gains an optional ceiling rope on the torso and exterior eye presentation. This is an executable test configuration, not full resting-human or suspended-human qualification.

The rope is an ideal massless unilateral distance constraint. Its moving gap/Jacobian and visible endpoint are encoded on the existing GPU command buffer. Existing articulated contact owns the impulse; zero friction admits tension without tangential support. The rope wraps the retained full-skin contact callback and preserves the other floor contacts. The point fixture is an ideal reference attachment, not distributed harness, cloth, or rope-contact mechanics. Physical body mass remains 97.13195069110952 kg.

Generate the runtime source with `tools/build_human_rope_runtime_source.py --repo REPO --output rope.metal --visual-kernel REPO/matter/src/NumiHumanRestingRopeVisual.metalinc --visual-header REPO/apps/NumiHumanRestingRopeVisual.hpp`; configure the native application with `-DNUMI_HUMAN_RESTING_ROPE_SOURCE=/absolute/path/rope.metal`. The binary verifies the configured SHA-256 before runtime Metal compilation. The tested combined source SHA is `c9a7efb7a9d783ae4ec3705dd56b206919a6977af82c57118a0294a0b2fcb21e`.

The reference environment is:

```sh
export NUMI_HUMAN_ROPE_ANCHOR=0,0,0.8,0.8
export NUMI_HUMAN_ROPE_SITE=-0.05459734828352807,0.14110308292600787,-0.002797657669904901
export NUMI_HUMAN_ROPE_ANCHOR_FROM_SITE=1
```

Anchor xyz is an offset from the initial site when FROM_SITE=1, otherwise fixed world coordinates. Setup lifts the source root by 0.35 m, applies the existing loaded NHEQ projector, and gives the exact same Float32 pose to initial geometry and physics. This is construction followed by release; it adds no per-step root assistance. Use the retained launch script described below for all assets and existing respiration/cardiovascular owners.

The eye presentation retains the canonical BodyParts3D ocular surfaces in Organ. Skin shows the original irises plus source-face sclera windows and posterior lens-face backing. Derived faces reuse source vertices and retain source-face ancestry. Internal ocular surfaces and opaque corneas stay in Organ to avoid hiding the irises in this renderer. Colors and dark backing are presentation approximations; this adds no optical solver, pupils as anatomical meshes, blinking, physical mass, or blood compartments. The original face vertices are unchanged. The strict eye unit verifies absent, partial, ambiguous, retired and complete source inventories. `face.png` is a static accepted-state preview, not runtime qualification.

Eye source attribution: BodyParts3D, Database Center for Life Science (DBCLS), CC BY 4.0; [official source data](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/data-5.html). Mixed-source reference anatomy is not a measured individual.

## Retained execution

All execution used SSH Numi / Apple M6. Build013 strict ABI41 compilation passed against the retained runtime library, with 133 dependency-header pins. Binary SHA: `b65a5b1d90a5b3b3d5007a0641bbc4e9da50feb78da4f18cb0a05fb0c1ce078f`; caller SHA: `6c164408dda402feaf7ef6bfde6c0d5c38d990460a1f3703ee29ddc191a2963f`; Visual SHA: `796e0506286ddfbcb7ab6d25c594d3c0bca8140e29b104c2b54100cc50d642ec`. Pending intercostal RibField work is excluded from this publication.

Current strongest full-scene physics result (attempt006/build012): 32 accepted steps, 0.064 simulated seconds. Last-step rope force was 815.133750439 N, tangential impulse zero, other contacts inactive, root assistance force/torque zero. This force is the last substep impulse divided by its 0.002 s timestep, not an accumulated segment integral or settled equilibrium. Existing injected rejection tests reported unchanged body, muscle/tendon, circulation, respiratory/controller history and common presentation with replay equivalence.

The run then stopped at the whole-mesh presentation audit: bone semantic 51004 / stable 49 / FJ3229, four exact-zero-area triangles. Face 8798 has a pre-existing 13.43 nm edge in its remesh tail, below the 119.21 nm world-coordinate Float32 increment at z≈1.83 m. The failed audit is retained; thresholds are unchanged. This is not a successful two-second or sixteen-second suspension demonstration. Local repair and rerun remain required.

Exact build/launch/assets/logs/video/receipts remain on SSH Numi:

- `/Users/n/numi-human-rope-abi41-build-20261018-013/build-report-abi41.json`
- `/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001/attempt-006/launch.sh`
- The same attempt directory contains `invocation.json` and `native-run/` traces, movie and failure receipt. Earlier failed attempts remain alongside it.

The included compact reports distinguish GPU visual dispatch tests, static source-preservation checks and scene execution. Storage deduplication recovered 1,505,935,360 bytes by verified identical-file hardlinks, preserving every path and byte. No five-minute coupled qualification, physiological range qualification, clinical validation or performance qualification is claimed here.
