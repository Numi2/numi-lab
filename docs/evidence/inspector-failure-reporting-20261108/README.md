# Inspector reports unrecoverable launch failures

The existing main-branch inspector worker logged launch failures, then asked
AppKit to terminate normally, which returned exit 0. The baseline native M6
run retains an owner-pack loading error and that false-success exit. Its pack
was authored by a newer integration revision and did not load on this branch;
that is a deliberately invalid launch, not a completed scene.

The repair preserves clean user-window closure and returns exit 1 when the
inspector worker reports another error. It does not change simulation or
contact options. The native probe uses a separate project, a synthetic Franka
world and matching owner packs generated against this branch, and the same
synthetic visual mesh in both arms. An absent camera parent returned 1 with the
expected diagnostic. With `panda_link7` as the camera parent, the command
returned 0 and its rollout reported eight steps. Both arms used the same
fixture, runtime binary, shared library and metallib hashes; input hashes were
unchanged. The probe outputs record exact argv and hashes. Synthetic assets and
the fixture generator are retained in `inputs/` for reproduction with
`tests/numi_inspector_failure_reporting_probe.py --help`.

This is a bounded launch and exit-status check. It does not qualify manual
interaction, physical accuracy, learning or performance. The scene, camera and
fixture are synthetic; the baseline and the two post-repair arms are distinct
records. The later unified Window has a different implementation and is
qualified separately on the integration branch.
