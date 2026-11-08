# Historical archive modifications

Added-path enumeration alone does not cover archive content modified later at an existing path. Repeat the reachable-history archive scan with added and modified path changes, all merge parents, no renames, no lazy fetching, and compare each resulting blob ID with the frozen branch-tip archive object inventory. Record missing blobs and errors; do not fetch or extract Human artifacts. Historical-only contents remain uninspected unless locally available and within a bounded read-only inspection.
