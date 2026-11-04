# Exact packed regional work
Eliminates region-by-largest-support tile grids for mode, block and fused FEM tangent builders. The immutable worklist is reused by direct dispatch, completed-work ranges and cached ICB paths. Padded outputs are initialized to positive zero once; their layout and the original summation tree are retained. No coordinate/model approximation or physical equation changed.

Direct GPU: 13 paired dispatches, bitwise (3 mode, 2 block, 8 tangent). Full solver: 917,050 checks. Native:32/32 accepted at15.625us;11/11 CSV traces and retained MRV captures bitwise against151.

Exploratory31-root uninstrumented GPU time: 325.330785s → 308.905352s (1.0532x throughput). One pair; initialization and instrumented root0 excluded. This is not a sustained or integrated performance qualification. Whole simulated prefix is0.0005s. The0.1x target and motion/resting qualification remain open.

Source snapshot was restored in memory and matched all39 execution pins. Source and binary/device/asset/configuration/command hashes are in the manifest and plan. Large geometry outputs remain on Numi under their exact run paths; their parity hashes are retained here. Routine logs/traces are compressed. Trial155 was disk-guard stopped before accepting a root; it is retained as a resource failure, not a physics failure.
