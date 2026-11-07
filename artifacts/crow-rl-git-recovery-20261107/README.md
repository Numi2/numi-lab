# Crow evidence recovery

This archive covers the exact 12,284 files listed in `manifest.json`: the American crow run tree, all 27 retained runtime trees, and 71 named build test directories. The authoritative root count is 73, including those two `.numi` roots. Source data total 5,881,893,915 bytes. Existing committed crow archives cover 11,899 files (3,965,058,718 bytes); only missing content is added here, in 275 content-addressed gzip chunks of at most 32 MiB uncompressed each. Stored new payload is 1,089,742,389 bytes. Duplicate source paths reference a single backing representation.

The manifest binds every original relative path, length, SHA-256, mode and capture modification time to an existing raw/gzip Git artifact or a sequence of new gzip chunks. This includes actual runtime binaries, libraries, shaders, sources and receipts. It preserves earlier failures and exploratory variants without promoting them to scientific qualification. Runtime compatibility still requires the matching device/OS and dependencies recorded in the original receipts.

Run from the crow worktree with its Python environment:

```sh
python/.venv/bin/python tools/crow_evidence_archive.py verify artifacts/crow-rl-git-recovery-20261107/manifest.json
python/.venv/bin/python tools/crow_evidence_archive.py restore artifacts/crow-rl-git-recovery-20261107/manifest.json --output /path/to/new-recovery --prefix .numi/retained-runtime/recurrent-rollout-admission-04
```

Verification reads committed Git objects, so bulk archives can remain absent from the working tree. Restoration writes a new destination and recreates the manifest's relative paths and executable modes. Omit `--prefix` to recover everything. No source is removed by capture, verify or restore. The separate prune command requires a pushed HEAD, verifies every unique content against Git, checks all source paths are unchanged and checks open handles before removing only manifest-listed files. Its receipt records the remote commit and actual `df` measurements.

Bulk evidence is stored in the repository, with per-worktree `skip-worktree` flags used to omit duplicate archive payloads from this checkout. This does not change shared Git history, objects or other worktrees. Essential policy and compatibility-shader inputs remain checked out. To materialize an omitted tracked file, run `git update-index --no-skip-worktree -- PATH` followed by `git restore --source=HEAD -- PATH`; retain its storage cost deliberately.

Future crow trials must use bounded temporary capture. Archive missing content once, commit and push, verify recovery, then remove the covered raw captures. Keep current build products and essential active inputs. Do not accumulate trial copies, runtime copies or checked-out archive payloads. Shared Git objects and other projects are outside this cleanup.

After pushing, `python/.venv/bin/python tools/crow_evidence_archive.py omit artifacts/crow-rl-git-recovery-20261107/NEW-omission-receipt.json` verifies each archive copy against its committed Git blob, checks open handles, marks exact paths skip-worktree, and removes those checkout copies. Compact recovery metadata and essential policy/shader inputs are retained. Its receipt lists every omitted path and blob identity.
