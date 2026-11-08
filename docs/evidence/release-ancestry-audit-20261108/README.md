# Read-only unified release ancestry audit

At audit time, remote HEAD was `numisolver` at
`698dd34fb77ba93d4a82a826e548fbd28b735371`; remote `main` was
`bb58f524353dc90b84c327908b26aeaaea5194ba`; the unified research branch was
`d6ab6d4221dc8a43545c323cc659e760789be95d`. The separately tested source
provenance research branch ended at `c7493ac3ed0b3d25f1453e8965fa318f80911b2f`
and descends from unified, but had not been integrated into unified.

Unified descends from neither public branch. Its merge base with main is
`304fd4be5013effeb997813139c24fce811cd988`, with 2 main-only and 61
unified-only commits. Main's standalone kinematic fix is patch-equivalent to
the integrated fix; the follow-up reset/recontact probe, test and retained
binary evidence files are identical, while unified's report has 18 added
lines. The later integrated reset result has an explicitly retained inactive
cache mismatch. No assertion is made that the two branch tips are identical.

The merge base with default numisolver is
`f52e88cb0f0ff6acdb4d81c904929dd4f30576ef`; the two histories have 109
solver-only and 2,049 unified-only commits. `overlap.txt` lists 50 source
paths changed on both sides since that base, including Window, training,
rollout, policy packs, C API, renderer and MetalWorld. Shared paths identify
reconciliation risk; no actual merge was attempted, so they are not presented
as confirmed textual conflicts. The public branch also contains distinct
bag, measured-surface and other work that must not be discarded.

On unified, source `d9db7f9f` has a fresh sparse-source build and bounded
installed non-G1 synthetic training/selection run, with an independent CPU
producer/consumer input-binding check. Its scope does not include a full
checkout, visible UI, physical accuracy, learned control, or performance.
The separate provenance candidate `a182448ec` has a network-denied build,
source inclusion proof, installed audit, and bounded non-G1 workflow; its
provenance does not establish authorship or a build attestation. Existing
opaque external-owner state guards fail closed and do not provide cross-owner
snapshot restoration. Default visual assets, complete dependency identity,
NumPy/SciPy CMake tests, and the deferred manual Window interaction remain
open release gates. Experimental directional CCD and AL contact remain off;
this audit does not qualify or alter either option.

This was read-only analysis of source and Git ancestry; no merge, build or GPU
run was made for this audit. The records are a snapshot of the listed remote
refs, not a promise that those refs stay current.
