# Positive reference-gap activation

The opt-in reference self-contact activation policy now handles every strictly
positive conservative separating-plane lower bound. A lower bound at or below
twice clearance previously restored the full authored 10 micrometre activation
width. The new width is min(authored, max(2*clearance, lower_bound/2)).
The hard collision clearance remains 1 nm. A small lower bound does not establish
a sub-clearance measured closest distance.

Two captured Float32 source pairs reproduce the branch error. The old shader
fails the new regression; the new shader passes 81,670 checks, including
intersecting/zero-gap fallbacks, finite bounds, canaries and clearance.

In the matched full Human release trial, the initial contact RHS norm falls
from 3.5161e-4 to 1.7063e-14. Two active Newton solves (13 Krylov columns) meet
the mechanical and vascular residual thresholds. The terminal physical
certificate still rejects object 15: zero physical steps are accepted.
These are convergence diagnostics, not native-motion or throughput qualification.

Exact plan, build/source hashes, compact traces and the failed terminal record
are retained in native-001. The source checkout was dirty; the build manifest
records that explicitly. No geometry, anatomy mass, support ownership or
physical acceptance threshold was changed by this increment.
