# Kinematic target rollback

MetalWorld applied a kinematic target before checkpointing the scene at the control-step boundary. Consequently a rejected step restored the requested target rather than the previously accepted scene. A later resident step with no new target could then evolve from that changed scene.

The contact preparation kernel now checkpoints the reset-adjusted scene before applying the current step's target. Reset placement is unchanged. A successful step still consumes and publishes the target; a rejected step restores the checkpoint. This is a focused transaction repair and does not change either experimental CCD or contact law.

## Clean-base native comparison

On clean 304fd4be, the same resident executable was run against the original and checkpoint-fixed Metal libraries. The fixture first accepts 13 cached contact manifolds with a kinematic scene body at x=1 m, then submits a target at x=0 that causes contact-capacity status 4. On the original library, q/v and manifold headers, points and counts restore, but scene x remains 0; an uncommanded neutral continuation fails again. On the fixed library, q/v, scene and all those manifold fields restore byte-for-byte, and the neutral continuation matches a clean reference. Both arms accept a separate target-bearing control and publish its requested kinematic velocity. This is a narrow fixture and does not prove every input, rod, reset or coupled-system transaction.

A separate candidate-solver test with 33 cached manifolds and a status-11 frictional rejection also showed byte-identical q/v/scene/cache restoration, identical subsequent continuation without resubmitting a target, and accepted target application. That test is supporting evidence, not a qualification or promotion of its experimental solver.

[Retained raw native results, source patch, timestamps and SHA-256 identities](evidence/ccd-kinematic-rollback-evidence-20261107.tar.gz) have archive hash f943101684f88387c28e5a75fbe8db544a0f34bc3c5c7d29c2f2d5a1195ab437. Exact executables and metallibs remain host-only in the evidence directories. The regression gate was evaluated against the retained fixed output; the GPU executions were the underlying native comparison.

## Integrated research worktree

The standalone change was cherry-picked as `5ada6932` onto the isolated
window/owner research worktree. On that exact revision, a freshly built
`metalrobo.ccd.kinematic_rollback` CTest passed the same bounded capacity
failure, populated-cache restoration, clean continuation and accepted-target
control. The complete output, timestamps, and executable, metallib and source
hashes are retained in
[the integration evidence](evidence/window-integration-20261107/kinematic-rollback-integrated/).
The original clean-base comparison remains separately source-bound. This is
not broad rollback or unified-release qualification.
