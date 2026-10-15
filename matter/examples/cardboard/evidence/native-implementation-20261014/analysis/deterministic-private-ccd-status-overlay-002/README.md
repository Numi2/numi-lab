# Deterministic private deformable narrowphase status selection

This is an isolated candidate patch against the frozen source109 snapshot named in `manifest.json`. It changes only the deformable contact narrowphase and its runtime setup.

Private FEM Armijo narrowphase runs now store failure tuples in per-candidate status records. A single reducer thread per environment scans only the active candidate prefix in ascending slot order and publishes the first failure before contact compaction. This removes GPU thread-arrival ordering from the reported private narrowphase failure. Earlier environment failures retain precedence. Normal production narrowphase passes selector 0 and continues using its existing environment-level first-writer status mechanism.

The ordering guarantee is slot-based: for a fixed generated candidate sequence, the lowest failed candidate slot wins. This does not claim a geometry-independent pair order when upstream candidate enumeration changes. Contact geometry, tolerances, acceptance, and candidate capacities are unchanged.

The additional scratch allocation is `environmentCount * deformableContactCapacity * sizeof(NMMatterStatusGPU)`. The unchanged ABI declares `NMMatterStatusGPU` as 48 bytes. At the current admission maxima of 4 environments and 131072 candidates per environment, the added scratch is 25,165,824 bytes (24 MiB); the allocation uses the existing checked `privateScratch` path and is charged to `residentBytes`.

`contract_test.py` exercises minimum-slot selection independent of simulated arrival order, preservation of earlier failures, active-prefix and empty-prefix behavior, the normal/private selector split, kernel writer coverage, and runtime dispatch order.

The CPU contract suite passes 9 tests. Metal compilation and native GPU execution were not run for this overlay; this is not native-qualified. Apply the unified `overlay.patch` only to the source109 base whose input hashes appear in the manifest.
