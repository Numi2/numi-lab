# Source 235-007 — contact sample compaction

This corrected snapshot preserves the 235-006 source tree while fixing its source binding. The inherited frozen map contained seven stale source hashes and omitted the newly added native checker host and Metal files. 235-007 binds 832 selected/support source entries, and the independent configure-support map binds 1,034 entries. Both maps were checked against file bytes; all entries match.

The snapshot adds NM_HUMAN_SUPPORT_TRACE_ROOT and NM_VASCULAR_TRACE_ROOT to the capture-mode exclusions. The sample-page optimization remains limited to the guarded pure-FEM free-BOX/capsule layout. Unsupported or diagnostic layouts do not enter the compact path.

A fresh source-bound build completed for numi-matter-cardboard-probe and numi-matter-cardboard-contact-sample-compaction-check. The build record binds 833 source files and five output artifacts. The checker was compiled only; no CTest, native checker, GPU, or shipping-world run was performed. The 235-006 source snapshot and its old build attempts remain unchanged.

The zero-byte build-cardboard-native-worker/worker.lock is retained because it is listed in the snapshot's configure-support map. It is a historical source-support placeholder, not an active lock.
