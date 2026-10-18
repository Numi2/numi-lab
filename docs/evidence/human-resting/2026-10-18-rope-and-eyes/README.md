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

Eye source attribution: BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International; [current official license](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html), [source mesh downloads](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html). Historical source receipts retain their original license declarations; source-specific rights remain attached. Mixed-source reference anatomy is not a measured individual.

## Retained execution

All execution used SSH Numi / Apple M6. Build013 strict ABI41 compilation passed against the retained runtime library, with 133 dependency-header pins. Binary SHA: `b65a5b1d90a5b3b3d5007a0641bbc4e9da50feb78da4f18cb0a05fb0c1ce078f`; caller SHA: `6c164408dda402feaf7ef6bfde6c0d5c38d990460a1f3703ee29ddc191a2963f`; Visual SHA: `796e0506286ddfbcb7ab6d25c594d3c0bca8140e29b104c2b54100cc50d642ec`. Pending intercostal RibField work is excluded from this publication.

Current strongest full-scene result is **attempt020/build013: 8000 accepted steps, 16.00000076 simulated seconds in 82.1152385 wall seconds (0.19485× real time)**. The continuous native scene completed three breaths and 19 cardiac filling/ejection cycles. Body and physiology clocks matched. Root assistance force/torque and other contact impulses were zero; the torso rope alone supported the body. Last-substep tension ranged 805.48–1039.62 N, with zero tangential impulse. These samples are impulse divided by 0.002 s, not accumulated segment integrals or a proof of settled equilibrium.

All 252 presentation audits checked 5,251,875 triangles with zero zero-area or nonfinite triangles. Eight accepted-geometry captures passed existing receipt/hash validation and finite-coordinate checks; their accepted body poses put the 0.8 m rope within 3.13 micrometres of its constraint. Maximum functional-volume relative error was 2.5742e-5; represented Float32 mass stayed 97.1319503784 kg. The maximum blood accounting error was 0.006985 mL; oxygen/CO2 balance errors were 0.000233/0.000466 STPD mL. These are numerical consistency observations, not physiological or anatomical intersection qualification.

Existing injected rejection tests passed: rejected work did not advance body, muscle/tendon, circulation, respiration/controller or common presentation state, including rejected multistep suffixes. Fresh-context reseed/replay matched the uninterrupted accepted prefix. Same-context retry remains unsupported; see [rollback log](native-16s/rollback-log-excerpt.txt).

[Native recording](native-16s/rope-suspension-reference.mov) is continuous and unretimed (84.195 s recorded duration). [Terminal frame](native-16s/rope-terminal.png) shows the 16.00 s endpoint. [Checkpoint report](native-16s/checkpoint-report.json), [capture verification](native-16s/capture-verification.json), compact CSV traces, compressed original log and exact [invocation](native-16s/invocation.json) are retained here. The [face preview](face.png) is a separate static source inspection.

Earlier attempts remain retained, including attempt006's bone49 precision collapse at 0.064 s and subsequent paired intercostal, passive-head and muscle precision failures. No audit threshold was relaxed. Existing constructors retired source edges below the Float32 emission scale; paired rib/intercostal repairs preserve shared coordinates. Index-only muscle repair is admitted only when allocated vertex fields, weights, bindings and identities are byte-exact. Two unmapped extensor-digitorum source vertices required bounded 0.377/0.262 micrometre separations with source-coordinate aliases moved together and incident orientations preserved. The working back and unrelated muscle rows remain exact. [Repair instruments, full compressed ancestry and report hashes](precision-repairs/source-index.json) document the modifications. Final tendon recompilation preserved all 832 endpoint records and coverage; only its embedded bone SHA changed.

Exact build, assets and full accepted geometry remain on SSH Numi:

- /Users/n/numi-human-rope-abi41-build-20261018-013/build-report-abi41.json
- /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001/attempt-020/native-run/
- Final paired rib/intercostal receipt: paired-rib49-intercostal-precision-009/resting-anatomy-receipt.json beside the attempt directory.
- Final muscle payload: extensor-source-precision-separation-008/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue beside the attempt directory.

This is a successful short native suspension checkpoint. Known hand/finger skin defects remain deferred. No blink mechanics, distributed harness, all-surface intersection qualification, five-minute baseline/intervention, physiological range qualification, clinical validation or real-time performance qualification is claimed.

## Storage

Verified hardlink deduplication preserved every existing path and content. The first two manifests account for 2,427,840,592 logical duplicate bytes; [third manifest](storage-dedup-003.json) retired another 3,894,845,440 allocated duplicate bytes across 103 generated files. For the third pass, observed free-space change was +3,616,301,056 bytes; APFS and concurrent OS activity make this distinct from the allocation sum. Numi had approximately 26 GiB free at final inspection. No sole source, current capture or recovery evidence was deleted.

## Fresh replay outputs

Run tools/launch_human_rope.py INVOCATION --output NEW_DIRECTORY --exclusive-wrapper /Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py with the qualified Numi Python. It verifies declared asset hashes, the active binary and configured rope source, remaps output paths into a fresh directory, rejects escaping paths and existing outputs, records launcher/wrapper hashes, and uses the existing exclusive-job wrapper. --verify-only checks preparation without launching physics. See launcher-smoke.json; these are launcher checks, not simulation evidence.

Historical attempt006 retains an obsolete build011 asset-map entry while its actual argv and binary hash identify build012. Its initial-pose description omits the loaded NHEQ projection; the execution log records it. Its failure-receipt environment points into attempt003. The historical record is retained with these limitations; the fresh launcher correctly rejects it. Attempt020 pins the actual binary, projected initialization and its own output paths; it passed fresh-launch verification and the native run.

rib49-repair/ retains the local repair instrument, compiled-face ancestry and source-preservation report. It retires two faces around a 13.43 nm source edge, preserving all allocated positions/normals and other bone index streams. Its early ad-hoc manifest is not the owner-compiled native asset. The existing precision constructor reproduced the same payload bytes exactly; composed-owner validation and paired tendon emission subsequently passed and attempt020 uses the final paired precision repair. The report candidate-manifest hash predates the frozen metadata correction; payload SHA remains 01c39cd8da439e2fed0770ff5d67e7028ea0d98d18a453721c26911dfd42d6bd.


## Replay the accepted checkpoint on SSH Numi

From the retained checkout, run with a new output directory:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3 \
  /Users/n/numi-lab-native-replay-publish-20261017/tools/launch_human_rope.py \
  /Users/n/numi-lab-native-replay-publish-20261017/docs/evidence/human-resting/2026-10-18-rope-and-eyes/native-16s/invocation.json \
  --output /Users/n/numi-human-rope-fresh-replay \
  --exclusive-wrapper /Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py
```

Add --verify-only for an asset/source preflight without launching physics. It is not native-run evidence. The original invocation pins all retained assets and runtime sources; it requires those assets on Numi. The source checkpoint is Human 40c2d287 and Lab 6fbcb759, with the binary's frozen source/header pins in the retained build report. This evidence publication changes no runtime source.
