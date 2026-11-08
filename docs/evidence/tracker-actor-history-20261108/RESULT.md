# Tracker actor-history physical-step restoration

At code commit `22e628f5b`, the native tracker snapshots the task actor-history buffer immediately before its visual observation writes. After the physical substeps, it restores that snapshot for each environment whose world or applicable contact status failed, while retaining the writes for accepted environments. The backup is submission-local, only spans the declared per-environment actor history, and is bounded by the source buffer. The callback checks the matching step, dimensions and single consumption. The MetalWorld callback structure gained a field, so C++ clients must rebuild against matching headers, library and shader.

This is a narrower guarantee than full rollback: the snapshot is **after** generic task observation has run. It does not restore earlier generic task changes, published actor output packets, the shared host sensor cadence, cached renderer frames, recurrent policy state, or effects of a partially failed GPU command. Automatic reset on a physics error prevents this fixture from proving matched continuation without reset. Tracker/renderer users must serialize in-flight use. The fixture's simulator-provided instance IDs are oracle perception, not deployment-valid detection. There is no performance claim. Experimental directional/interval-local CCD and augmented-Lagrangian contact remain disabled.

## Evidence

The preregistered [plan](PLAN.md) uses a 30-step, two-environment Franka scene with contact capacity one, CCD disabled, four substeps at 0.02 s per control step, and deterministic 35% masked-depth dropout. The wrapper in [probe.cpp](probe.cpp) captures the public actor-history buffer immediately before the tracker callback, after its writes, and after the physical-step callback. The control uses the previous tracker-history repair with a read-only diagnostic accessor; its source qualification is retained in the [earlier report](../tracker-step-rollback-20261108/RESULT.md). The candidate uses the actual main runtime; integration uses the integrated runtime. The accessor is not needed by this actor-history capture.

Five of 60 environment steps fail: `(13,0)`, `(14,0)`, `(26,0)`, `(27,0)` and `(27,1)`; the remaining 55 are accepted. At every failure, candidate post-step bytes match the pre-visual snapshot; at every accepted step they match the post-visual snapshot. The old runtime retained the visual writes even for failures. In mixed steps 13 and 26, both environments had changed visual history: the failed environment was restored and the successful one retained its update. This directly measures the boundary; it does not establish that all state was restored.

The old, candidate and repeated-candidate `q`, `v`, scene, actor and status captures are byte-identical, as are the two candidate histories. The integration build's six captures (including history) are byte-identical to the candidate. [analyze.py](analyze.py) checks the paired results, including the old failure. Saved logs and raw arrays accompany this report.

Both main and integrated builds passed all 12 targeted native tests, including direct mixed-status tracker tests and 18 GPU restoration layout/status combinations with empty and nonempty object/depth and actor histories, sentinel bounds, world/contact failures and no-contact mode. Mismatched and repeated step callbacks are rejected. A host rejection after the physical-step callback followed by retry yields q/v/scene/actor/status bytes identical to its control. The legacy no-hook callback's retained rejected-submission retry matches its control; it was run before the final internal backup-size refinement, while the final native suite and after-step retry were rerun afterward. No stronger legacy requalification is claimed.

## Runtime identities

All runs on the Apple M6. Main source `22e628f5b`; integration source `c22932616`. SHA-256 values:

- Main library `2221b2a7559acc3c33743dbe8db1bcc7f0efa5c274df41adaa72f3ed8eae610a`, shader `ca4792eb20cc85eea64412ab075204a271b003f0b16575697a93c1aee4fd0fe7`, probe `4311261f7f628442ebe8bb1e14b0e4760e0132c515f056eeacc8805884cf5729`.
- Integration library `9d8b16844f4aa890d4726ce8e5cc82078b4fef7eaf56f55f19a15388cc48349f`, shader `05f25e711e60808d6b4f3e756fd9bb4da4f072541eb38882d7353ab21b82522b`, probe `09c449db907762262d418df42e79a343e8dfdd270485c495c8052e02802e30d1`.
- Old diagnostic library `64650cd689871f39efe8cedc3a0b7f5fb37ecdb5add94d58feca7f0ce95ee80c`, probe `e1273b4343d5befb1ad12ff7e962a04143a896ef64a30752bc142c63c13e270c`.

Further work must make sampling and relevant actor/policy state follow per-environment physical acceptance, and test actual continuation and command failure. This evidence does not certify that larger transaction.
