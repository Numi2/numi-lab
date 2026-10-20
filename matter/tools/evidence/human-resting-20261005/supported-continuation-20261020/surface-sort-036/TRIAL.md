# Measured surface-sort synchronization036

Profile031 measured100 calls of the existing stable primitive sorter at about
10.035ms/call, the largest of the three explicitly sampled contact kernels.
Preserve its ten3-bit stable radix passes, array/layout/dispatch contracts,
active primitive filtering, cache skip semantics and all physical equations.

Candidate: carry compaction prefix identically in each SIMD lane, publishing
once before radix reads. During each scatter tile compute all bucket outputs
before updating any cursor, then advance the eight independent cursors with
two shared barriers instead of two barriers for each of eight buckets.
No new buffers, state ownership, contact law, force or tolerance.

Before replacing the running baseline: retain its exact Metal library/hash.
Run the direct production sorter fixture against that library and candidate:
active counts, key/index permutation, monotonic ordering, stable equal-key
ties, odd tails, empty/mixed/all-active and cache controls must pass. Compare
their deterministic raw outputs byte-for-byte.

Run the same two-step enabled-block full human profile sequentially before
and after the shader change. Require exact accepted CSV row parity first.
Retain selected per-kernel dispatch counts/timestamps, source and binary hashes.
Treat this as a bounded root1 cost comparison; startup-dominated process wall
time is not a steady-state real-time factor or a whole-human speedup claim.
Only one GPU job/build at a time; source-only preparation overlaps035.
