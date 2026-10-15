# Bounded preconditioner-only diagnostic

Overlay based on frozen source066. `NM_FEM_REGIONAL_RAP_CAPTURE_ONLY=1` retains the selected regional RAP/candidate/history/cache and optional graph-factor sidecar while omitting finite-difference readback and encoder work. The aggregate shared readback cap stays 32 MiB. Existing private FD scratch admission/allocations are still present; this is not a claim of eliminating all FD scratch.

Graph metadata hashes the exact compiled host topology and immutable mask passed to the production builder: little-endian u32 node count, u32 tet count, then row-major u32 tet connectivity; raw u8 mask, 1=free and 0=fixed. This identifies actual global-node topology; matching synthetic allocation counts cannot substitute for it.

Strict runtime syntax passes after adding the frozen-source include root. The first missing-include invocation is retained. Native observational parity remains required. Owner review found two integration defects retained in source067: the pair comparator expects a bare FD file instead of regional sidecars in this mode, and mode1 can silently miss capture if selector/path is absent. A follow-up overlay repairs those contracts before qualification. No production solver tolerance, operator, state or acceptance rule changes here.
