# FEM full-lane admissibility trial 030

Source 67c85cb3 retains the shared VF contact witness and FGMRES budget64 from
029. The FEM backtracking certificate now reduces every lane before electing
the writer, and predicts the same fixed/reference-displacement node updates
as nm_fem_apply_solution.

The old production shader missed an invalid element on thread57 and selected
alpha1/minJ1. The repaired shader must select alpha0.5/minJ0.656964 for that
fixture, while fixed-node and valid controls select alpha1; repeat in both
world-position and reference-displacement modes. Require the unified
preparation/geometry/contact/interface checks and two-step rollback first.

Request32x125us from the identical scene and assets, keeping material validity
J>=0.35, timestep, forces, mass, accepted-state ownership and all original
physical acceptance checks unchanged. Old contact trace is disabled.
More than22 accepted roots passes the last029 failure point; all32 roots
are required to pass this bounded4ms horizon. Retain any failure.
This is a numerical integration test, not static support or physiological
qualification.
