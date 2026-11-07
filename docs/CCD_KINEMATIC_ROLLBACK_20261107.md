# Kinematic target rollback

MetalWorld applied a kinematic target before checkpointing the scene at the control-step boundary. Consequently a rejected step restored the requested target rather than the previously accepted scene. A later resident step with no new target could then evolve from that changed scene.

The contact preparation kernel now checkpoints the reset-adjusted scene before applying the current step's target. Reset placement is unchanged. A successful step still consumes and publishes the target; a rejected step restores the checkpoint. This is a focused transaction repair and does not change either experimental CCD or contact law.

## Clean-base native comparison

On clean 304fd4be, the same resident executable was run against the original and checkpoint-fixed Metal libraries. The fixture first accepts 13 cached contact manifolds with a kinematic scene body at x=1 m, then submits a target at x=0 that causes contact-capacity status 4. On the original library, q/v and manifold headers, points and counts restore, but scene x remains 0; an uncommanded neutral continuation fails again. On the fixed library, q/v, scene and all those manifold fields restore byte-for-byte, and the neutral continuation matches a clean reference. Both arms accept a separate target-bearing control and publish its requested kinematic velocity. This is a narrow fixture and does not prove every input, rod, reset or coupled-system transaction.

A separate candidate-solver test with 33 cached manifolds and a status-11 frictional rejection also showed byte-identical q/v/scene/cache restoration, identical subsequent continuation without resubmitting a target, and accepted target application. That test is supporting evidence, not a qualification or promotion of its experimental solver.

[Retained raw native results, source patch, timestamps and SHA-256 identities](evidence/ccd-kinematic-rollback-evidence-20261107.tar.gz) have archive hash f943101684f88387c28e5a75fbe8db544a0f34bc3c5c7d29c2f2d5a1195ab437. Exact executables and metallibs remain host-only in the evidence directories. The regression gate was evaluated against the retained fixed output; the GPU executions were the underlying native comparison.

## Simultaneous reset and rejected target

A separate resident fixture first seeds 13 manifolds, then requests a reset to scene x=2 m and a target at x=0 that triggers capacity status 4. Compared with an independently accepted reset-only control, the failed path has byte-identical q, v, scene and manifold count (zero). Immediately after rejection the full allocated header and point buffers differ, so the complete-buffer equality test fails. This is not described as byte-exact cache restoration.

Reset sets the live manifold count to zero and checkpoints only entries below the count; publication copies the full allocated buffers. The collision manifold lookup bounds old entries by that count and dereferences them only for a matching active entry. In this fixture direct recontact after the reset/rejection creates 13 manifolds, and the entire resulting q/v/scene and manifold buffers match the clean reset-only control byte-for-byte. This supports the live-count boundary for the tested path, but does not prove every consumer ignores inactive storage.

The first reset setup failed host validation because reset-state spans had no reset mask; it is retained. The corrected cases and field-by-field mismatch are in [ccd-kinematic-reset-evidence-20261107.tar.gz](evidence/ccd-kinematic-reset-evidence-20261107.tar.gz), SHA-256 88f389a8ad2b9f7c39c5f5c02574bd2d4ea01809db70420b3ba0c1c63e9c32bf. Exact executables and metallibs remain host-only.

After the scheduled test, an inadvertent extra native gate run occurred between 22:54:07 and 22:54:15 UTC; exact process timestamps and its full output were not separately retained. The copied probe SHA-256 was 9467ad2b7f046c6daeef586bfedc9706e93dc5a631a0eee561dfcb9adba924a4 and the fixed metallib SHA-256 was d7d37bce197e77e3056ba00abbba303d4c96cbf7fc6eeb95a7f765d101380866. Other active owners reported no overlapping GPU runs in that window. No performance claim relies on it.
