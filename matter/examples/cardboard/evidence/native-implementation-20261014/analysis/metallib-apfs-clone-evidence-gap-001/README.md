# Metallib clone-compaction evidence gap

This is an explicit evidence-gap package, not a reproducible proof of the earlier APFS clone-replacement operation.

A search of the task analysis tree and `/private/tmp` did not recover the operation script, its complete per-path before/after inventory, or its receipt. The prior execution notes reported 378 clone replacements, zero reported errors/skips, and a stop after the free-space threshold was crossed. They recorded `df` available space as 1,665,916 KiB before and 2,623,616 KiB after (a 957,700 KiB difference, which may include concurrent cleanup). Those are transcript-reported values only. The exact 378 paths and the historical per-path bytes, metadata, open-handle checks, and clone semantics are unverified here.

`current-inventory.json` is a fresh, read-only census. It hashes every current `.metallib` under `build-cardboard-provenance-*`, excluding build directories 154 and 157 from the in-scope set while separately inventorying their four current files. It records current stat fields, xattr listing digests/text, ACL listing lines, and confirms no non-unit link counts. The current SHA groups have no mismatch in size, mode, uid/gid, flags, xattr listing, or ACL listing. This does not identify which files were replaced earlier or prove their former contents/metadata. The lsof query returned code 1 with empty stdout/stderr, consistent with no matched open files at census time.

The census was run with `/Users/home/.pyenv/versions/3.10.12/bin/python3 census.py`. Current free space was 2,428,840 KiB before and 2,427,452 KiB after this read-only census. No further files were replaced as part of this package.

Do not publish this package as an independently reproducible compaction proof or use its current-state inventory as the missing historical before/after manifest.
