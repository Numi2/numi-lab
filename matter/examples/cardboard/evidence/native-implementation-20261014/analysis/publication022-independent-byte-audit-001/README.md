# Publication reference 022 independent audit

Read-only audit of the 57 newly published cases and 1,184 archived files. The script verifies source and archived hashes, raw copies, streamed gzip payloads, summary-reference021 prefix preservation, exact pathspec coverage, and the zero-staged-path guard. It does not run the archive builder, stage files, or mutate the archive.

Cleanup measurements are advisory only. Snapshot `st_blocks` are reported as filesystem allocation, not a prediction of physical bytes returned on APFS because cloned/shared extents may be present. The restore bundles are currently untracked, so they are not yet a durable retirement basis.

Run: `/Users/home/.pyenv/versions/3.10.12/bin/python3 audit.py`
