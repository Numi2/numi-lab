# Closed generated CMakeFiles compaction 031

This package contains lossless, per-tree gzip/PAX tar archives for the 160 real, previously unarchived, nonempty `CMakeFiles` trees under `/Users/home/numi-cardboard-20261006/build-cardboard-provenance-023` through `-153`. The two supported locations are each build's `CMakeFiles` and `matter/CMakeFiles`. Build `027` is deliberately excluded; build suffixes above `153` and `current` are outside the scope. Fifty-five in-scope roots that already had `RESTORE_FROM_COMPACTION.txt` pointers from compaction-001 were left untouched; empty trees were also left in place.

For each of the 160 targets, the package records a full per-entry inventory with SHA-256, mode, uid/gid, nanosecond mtime, flags, link count, xattrs, and ACL listing. The file inventory includes 164,568 files and 40,376 directories, totaling 427,607,319 logical file bytes and 989,032,448 allocated file bytes. The archives total 94,806,147 bytes. All trees carried `com.apple.provenance` xattrs (204,944 entries total); bsdtar used PAX with `--mac-metadata --xattrs --acls --fflags`. All 160 ACL listings had zero entries, and no symlinks, special entries, nonzero file flags, or hard-linked files were found.

Before each removal, the workflow checked that the target had no Git-tracked paths, active build/probe processes referencing it, or open file handles. It created one tar.gz archive and one compressed inventory, extracted the archive into a temporary directory, reapplied exact uid/gid/mode/nanosecond mtime, and independently compared the restored path set, every file hash, xattr values, ACL listing, flags, and link counts against the original inventory. Only after that restore check passed did it re-inventory and re-hash the original, repeat the process/open-handle/tracked-file gates, write the recovery pointer, and unlink exactly the inventoried files/directories. The original birthtime is recorded but changes on archive extraction; ctime is naturally new. No build inputs, source snapshots, source files, executables, metallibs, static libraries, or data outside those exact `CMakeFiles` paths were removed.

The preflight `df` reading was 600,736 KiB free; the first package final-audit `df` reading was 1,524,824 KiB free, an endpoint increase of 924,088 KiB. The sum of per-tree before/after readings is also retained, but it differs from the endpoint delta because the shared filesystem changed during the work; neither number is asserted as a sole causal attribution. The 2 GiB target remains unmet by about 559 MiB, so no additional scope was touched. A later independent `df` reading may differ.

## Restore

Restore any one tree only when no build is using that tree. The command checks for build processes and open handles, verifies the archive and inventory digests, extracts to a temporary directory, reapplies metadata, verifies every hash/path/attribute, then atomically replaces the pointer-only `CMakeFiles` directory:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/closed-build-intermediate-compaction-031/compact_trees.py restore build-cardboard-provenance-137__matter
```

Use any key listed in `preflight.json` or `package-manifest.json`. Archives and inventories are the recovery source and must remain available. The exact script used during compaction is preserved as `compact_trees-operation-v1.py`; `compact_trees.py` is the restore-safe version with the additional pre-restore process and `lsof` gates.

`final-audit.json` and `audit_package.py` record the read-only final archive/pointer consistency check. `batch-progress.jsonl` is a progress log only; its `elapsed_s` field was computed from an absent timestamp and must not be interpreted as a duration. Per-tree receipts contain the authoritative measured `df` readings.
