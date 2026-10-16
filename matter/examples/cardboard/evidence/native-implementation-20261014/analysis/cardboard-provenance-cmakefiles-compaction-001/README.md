# Provenance build CMakeFiles compaction 001

This package archives only generated `CMakeFiles` subdirectories in 56 closed legacy build trees under `/Users/home/numi-cardboard-20261006/build-cardboard-provenance-*`. Every tree was inventoried and checked for Git-tracked files, active build-process references, and open handles before archiving and again before removal. Tracked-file count and open-handle count were zero. Each target was restored from its archive into a temporary directory before source removal; file SHA-256, exact path set, mode, uid/gid, nanosecond mtime, flags, link count, xattrs, and ACL listing were checked. The inventories record birthtime, but the filesystem assigns a new birthtime on archive extraction; no other metadata mismatch was accepted. Trees with ACLs, symlinks, special entries, hardlinks, or nonzero flags fail closed.

The archive files, compressed per-tree inventories, prepare/removal receipts, recovery script, and this manifest are the retained recovery path. Restore one compacted tree with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/cardboard-provenance-cmakefiles-compaction-001/compact_one.py restore build-cardboard-provenance-085
```

Replace the final build name with any key in `archive-manifest.json`. Restoration refuses unexpected target contents or changed archive/inventory hashes, verifies extracted metadata and contents, then replaces the pointer-only `CMakeFiles` directory. Restore trees sequentially so temporary extraction has adequate free space.

The explicit exclusions were `build-cardboard-provenance-027`, `-154`, `-157`, and any `current` path. The external 027 case was not inspected or changed. No build source, executable, metallib/shader, static library, `compile_commands.json`, or content outside these exact `CMakeFiles` paths was removed. There were 64 other eligible `CMakeFiles` trees remaining when the shared free-space target was reached; they were left in place.

At the start, the observed free-space value was 613,097,472 bytes. A recorded post-removal reading crossed the 2.5 GiB target at 2,686,521,344 bytes (target 085 receipt). Later readings fluctuated around 2.5 GiB while a separate task performed a temporary restore and clone compaction. The aggregate per-tree df deltas are not attributed as sole causes of global free-space changes. Stop condition was the observed shared 2.5 GiB threshold; no further target directories were compacted.

The first 023 removal receipt has a field-label error: `df_before_remove` contains the post-removal reading. Its separately recorded `measured_free_increase_since_prepare` is computed from the prepare-time free-space value and post-removal reading and is retained. Do not use that one `df_before_remove` field for analysis.
