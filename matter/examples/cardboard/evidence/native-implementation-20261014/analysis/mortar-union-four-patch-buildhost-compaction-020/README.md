# Closed build-host compaction for overlays 004 and 005

This package contains lossless ZIP archives of only the generated `build-host` directories for mortar-union-four-patch overlays 004 and 005. The original build outputs are reproducible compile-only artifacts; overlay sources, patches, manifests, logs, and all non-build-host data remain in place.

Overlay 004’s manifest-bound executable is inside the 004 archive and was rehashed there after compaction. Overlay 005’s manifest-bound `build-output.txt` is outside `build-host` and remains in place. The archives were made with macOS `ditto` preserving extended attributes and ACLs. Both source trees had zero ACL entries. Restoration was tested by extracting each archive, applying exact nanosecond mtimes from `preinventory.json`, and matching all inventoried relative paths and file SHA-256 values, sizes, modes, uid/gid, mtime_ns, flags, link counts, and extended attributes. Birthtime is filesystem-created on extraction and changes; it is not restored. For 005, `ditto` extraction produced an empty `CMakeFiles/CMakeScratch` directory absent from the source and ZIP listing; the test removed it only after confirming it was empty. The restore program enforces the same rule and rejects any other extra path.

If a `build-host` directory contains only the package-created `RESTORE_FROM_COMPACTION.txt` pointer or a partial deletion whose remaining files still match the original inventory, restore it with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-buildhost-compaction-020/restore_build_host.py 004
python3 /Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-buildhost-compaction-020/restore_build_host.py 005
```

The program verifies the archive and pre-inventory hashes, refuses unexpected or modified contents, extracts to a temporary directory, reapplies exact modes/ownership/mtime, checks hashes and extended attributes, then replaces the pointer directory with the restored build tree. The ZIP archives and pre-inventory are the recovery source and must be retained.
