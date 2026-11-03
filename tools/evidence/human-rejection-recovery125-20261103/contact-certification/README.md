# Reference contact certificates and completed Newton work

The complete Human release scene converges mechanically in two active Newton
solves but its terminal certificate rejects contact. This increment preserves
the physical residual/material/contact owners and fixes numerical and work
selection inconsistencies.

- Iterate the active deformable contact prefix, validating its indices, rather
  than the four-million-slot allocation.
- Use per-contact scale for the legacy nonreference gate. Reference FEM features
  can instead pass the existing compensated separating-plane certificate at
  the unchanged 1 nm ordinary clearance.
- Supply feature-derived candidate axes when a rounded closest-point normal
  fails to separate the complete convex features. Every candidate axis still
  needs an outward-rounded interval proof; approximate axes alone never pass.
- Refine only uncertain projection intervals with an exact Float32 expansion.
  TwoProduct/TwoSum guards conservatively fail on uncertain underflow/overflow.
  The fixed expansion holds the 24 terms of the exact dot product.
- Make the existing initial-touch/nonclosing policy consistent at the terminal
  certificate. An initially positive, sub-clearance feature pair may remain
  exactly unchanged or open. Opening is proved by a candidate distance lower
  bound exceeding an initial convex-witness distance upper bound. Closing,
  newly sub-clearance, intersecting, invalid and nonfinite cases still reject.
  This is an explicit initial-contact exception, not a claim that all source
  gaps are at least 1 nm.
- Skip completed Newton contact limiter work; tiled lanes publish neutral bounds
  rather than retaining stale scratch.
- Add bounded terminal diagnostics (less than 16 KiB total), including one exact
  failing contact and both endpoint features per environment. It is opt-in and
  does not advance or modify state.

## Verification

82,445 native GPU checks pass on SSH Numi. They cover the captured 483 nm edge
pair, the captured 0.687-to-0.699 nm opening triangle contact, ordinary
clearance, closure/crossing, initial intersections, sparse/empty active sets,
poisoned padding, invalid indices, inactive serial/tiled paths and canaries.
Retained failed check logs show why the axis-only and unrefined-interval
approaches were insufficient.

Matched stage timing for the limiter convergence guard:
50.2881 -> 24.8568 seconds summed GPU stages. Completed-iteration globalization
falls from 1.86708 to 0.04831 seconds per iteration. Both endpoints still reject
the same physical transaction. With the existing tiled limiter and removal of
per-contact trace readbacks, the next trial totals 21.0781 GPU stage seconds.
That combined comparison does not isolate the tiled kernel's individual gain.
These are initialization diagnostics, not sustained throughput or 0.1x.

The source checkout is dirty. Trial manifests retain actual source hashes and
patch hashes, binary/metallib pins, exact arguments and environment. Geometry,
mass ancestry, source identities and support ownership are unchanged.

The nonclosing-native-002 full scene accepts its first 15.625 microsecond
physical step. Its second step opens the same feature gap further but fails a
Float32 axis certificate. The next tested increment carries the feature plane
axis in compensated words; both captured opening endpoints pass, while their
reversed closing endpoints reject. Its full-scene trial is retained separately.
The four-Newton diagnostic configuration totals 7.3089 GPU stage seconds at
root 0. This is not a matched isolated-kernel comparison or sustained throughput.
The first 16-step launch (nonclosing-native-001) failed an inherited geometry
capture cadence setting before physics; its corrected launch is 002.

The compensated-axis native trial accepts two consecutive physical steps
(31.25 microseconds total). All first accepted coupled-trace columns match the
prior accepted-step trace exactly. The third step rejects with a global
nonlinear status and zeroed diagnostics. It does not exhaust the 64-column
linear budget (observed maximum five). A bounded globalization trace is required
before attributing that failure to a specific owner.
The root-0 stage sum with four Newton / 64 Krylov budgets is recorded in
combined-comparison.json. These are tested diagnostic configurations; no global
default budgets or acceptance tolerances are reduced. This remains an
unbalanced initialization smoke, not motion, supported rest, or 0.1x qualification.
