# Publication028 independent staged-byte audit

This read-only audit independently checked the source157 publication package against its external case inputs, repository copies, Git index, worktree, and the archived recovery pointers. The result is **PASS** for the checked bytes. It does not imply a completed native468 trajectory or physical qualification.

The summary contains 1,480 records; its first 1,460 records match `summary-reference027.json` exactly. The new 20 records contain 715 payload files: 707 byte-identical raw copies and 8 gzip copies whose decompressed bytes match their source hashes. The summary and publication binding match the declared SHA-256 values. Pathspec028 has 5,566 entries and its first 4,849 bytes are the exact pathspec027 prefix; the new suffix has 717 entries (715 payload destinations plus the old summary reference and binding).

All 787 source paths in source-build.json independently hash to the declared source157 values. The external recovery index and all indexed archive/inventory descriptors hash correctly; all 128 published restore-pointer copies match their source pointer hashes. The staging list contains 722 paths with the declared hash. Those staged index blobs match all worktree files byte-for-byte (51,488,434 bytes compared); there are no unstaged tracked changes or staged deletions. Four staged integration paths lie outside the evidence pathspec: implementation-20261014.md, solver-milestone-source157.md, solver-progress-20261014.md, and cardboard_probe.mm. Exactly two unrelated untracked files remain and neither is staged.

Command003's refusal is retained and its refused command SHA matches the refusal report. Command004 binds to the source157 executable SHA recorded by source-build.json; the external current-status note records that command004 was later launched and is active, while its solver output and launch guard remain outside this publication. The metallib operation's historical evidence gap remains explicit: the prior clone-replacement operation is not independently reproducible from retained receipts.

Audit lineage: audit001 records the original checker-side false negatives (misread qualification value, one-character expected pathspec hash typo, evidence-root prefix omission, and directory-vs-file untracked expectation). The unchanged 001 script was rerun to restore its deterministic failed report after its first output path was accidentally reused during 002 setup. Audit002 corrects only those verifier assumptions and adds independent hashing of the 787 build-source entries; it does not modify source, staged files, or publication artifacts.

Run with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/publication028-independent-byte-audit-002/audit_publication028.py
```

The publication still states that numerical convergence, physical assembly, analogue validation, and crush/impact prediction are unqualified or unexecuted. No physical qualification is claimed here.
