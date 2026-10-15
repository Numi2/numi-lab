# Closed CMake dependency-file retirement 016

Removed only regular generated `.d` dependency files below `CMakeFiles` in build-cardboard-provenance-133, -134, -135, and -136. The exact pre-removal file manifest, including SHA-256, logical size, APFS allocated bytes, device, inode, and mode, is in `before.json`. `after.json` records the post-removal state and source-build artifact verification. `summary.json` binds both manifests.

The operation removed 378 files: 56,150,316 logical bytes and 56,946,688 allocated bytes. Counts were 126 each in builds 133, 135, and 136; build 134 contained no `.d` candidates. The `/Users/home` filesystem available-byte count rose from 2,906,722,304 to 2,954,272,768 bytes, a measured increase of 47,550,464 bytes.

Immediately before deletion, `ps -Ao pid,ppid,stat,etime,command` had no command line containing any of the four selected build roots. `lsof +D <build>/CMakeFiles` returned no open-file rows for each of the four trees (exit code 1, empty stdout). The post-removal scan was also clear. All source-build artifact hashes and CMake metadata hashes recorded by each build's `source-build.json` matched both before and after: 133 (7 artifacts + 2 metadata), 134 (3 + 2), 135 (7 + 2), and 136 (7 + 2). The build receipts and products remain in place.

No source snapshots, `.o`, `.air`, `.metallib`, linked binaries, source files, output files, or histories were removed. Builds 137 and 139 were outside the candidate scope and were not scanned for deletion.

The one-shot action is recorded by `retire.py`; do not rerun it as a cleanup command. The evidence files are immutable records of the completed action.
