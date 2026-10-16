# Compaction follow-up v2

This supplemental seal preserves the original v1 `archive-manifest.json` unchanged and adds one previously eligible closed tree, `build-cardboard-provenance-084`. Its 1,595 regular files (8,925,184 allocated bytes) and 407 directories were archived, fully restored into temporary storage, and matched by path, SHA-256, mode, uid/gid, nanosecond mtime, flags, link count, xattrs, and ACL listing before the source tree was removed. The second pre-removal verification again found no tracked files, build-process references, or open handles. The tree now contains only `CMakeFiles/RESTORE_FROM_COMPACTION.txt`; its recovery archive and inventory are retained.

The per-target removal receipt recorded 2,684,616,704 bytes free immediately afterward, above the 2.5 GiB preflight. At v2 manifest generation the shared filesystem reported 2,680,762,368 bytes free (2,617,932 KiB), below 2.5 GiB by about 3.4 MiB. This is recorded without attributing the intervening change to any one task. No additional tree was compacted.

V2 covers 57 compacted trees and leaves the other 63 eligible trees untouched. It binds v1 by SHA-256 and leaves v1 intact. Recovery for 084 is:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/cardboard-provenance-cmakefiles-compaction-001/compact_one.py restore build-cardboard-provenance-084
```

The v1 package's stated boundaries remain: only inventoried files and directories below exact `CMakeFiles` paths were removed; build sources, executables, metallibs/shaders, static libraries, `compile_commands.json`, other build outputs, external case 027, and current/154/157 trees were untouched. Shared `df` observations may move independently of these per-target receipts.
