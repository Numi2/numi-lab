# Result: Franka asset prerequisite rejected

The pinned process ran 2026-11-08 00:02:34.823390–00:02:35.209317 UTC and exited 2; preflight and postflight hashes passed. The owner refused construction because the reused visual pack's symbolic rigid-body binding did not bind Franka's pick_object. No task step, direct-winner qualification, present-versus-missing admission, tracker snapshot, or recovery occurred. The zero-byte snapshot and error are retained.

This is a fixture asset mismatch, not evidence of a frame-status or rollback failure. A separate follow-up must use an explicitly authored, content-addressed visual pack with the correct pick_object binding; do not relabel this result.

