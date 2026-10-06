# Constitutive path resolution audit

This folder preserves a CPU-only, independent FP64 material-point check of the current Haj-Ali-inspired ideal Hill plasticity equations. It is a synthetic constitutive diagnostic, not a corrugated-board, finite-element, contact, or physical validation result.

`hill_material_point_oracle.py` solves the copied liner and medium laws in double precision using Python's standard library. It verifies that each archived paper-calibration material copy is equation-equivalent to its canonical material after comments and whitespace are removed, checks for explicit time/rate tokens, then evaluates 32, 64, and 128 subdivisions. The input material SHA-256 bindings are:

| Material input | SHA-256 |
|---|---|
| `paper-calibration-009/liner.nmatter` | `c6cd258db8516a7ef3a3a8c1f2482c1a32e1e8b972dd07dbf8b4fb6a9e9d758c` |
| `hajali2009_liner_hill_ideal.nmatter` | `787bf121f4ce359235e28691892e9d77adffde05faf10f2a90593d9f4b32398d` |
| `paper-calibration-009/medium.nmatter` | `d1665e8846f48bc366f0f3d50bb8eafb8c6d382dd0acf9d9c4f8f8848a3cfce5` |
| `hajali2009_medium_hill_ideal.nmatter` | `ecfd9de3f4550a6a269d7a6559781b16e15316cf3d587093f2e11a2e6942a8e8` |

The tested paths are:

1. Green MD compression `E11: 0 → -0.010` in equal increments, followed by the same number of fixed-strain updates.
2. Proportional Green MD compression plus engineering CD shear `(E11,C12): (0,0) → (-0.006,0.004)`, then a strain-controlled return to `(0,0)`. The endpoint is prescribed zero total strain, not force-free relaxation.

The local equations have no explicit `dt`, time, or strain-rate input. Twofold subdivision from 64 to 128 steps changed the largest tested endpoint vectors by 0.0787% in stress, 0.1125% in plastic strain, and 0.1776% in accumulated multiplier. Holding the MD-compressed point near yield for 128 updates accumulated smooth-complementarity drift: liner `Δλ = 4.6581e-9`, `||ΔEp|| = 1.6324e-8`, and stress drift `3.2275e-5 MPa`; medium `Δλ = 4.7437e-9`, `||ΔEp|| = 1.0883e-8`, and stress drift `2.0721e-5 MPa`. All oracle plastic multiplier increments were nonnegative.

The reported smooth Fischer–Burmeister regularizer is `1e-8` and the Hill norm floor is `2e-6`. The tiny hold drift is numerical pseudo-plastic flow from that regularization, not a time-dependent material response. These local path differences are much smaller than the parent-reported board-level temporal-refinement changes, but this material-point test cannot diagnose changes in board deformation history, contact, global Newton/line-search behavior, GPU local-root tolerance, or mesh response. It does not establish that the constitutive law caused or did not cause the whole-board difference.

The full raw path outputs and parent-supplied board comparison context are in [`constitutive-path-audit.json`](constitutive-path-audit.json). The JSON explicitly labels the board figures as supplied context, not CPU-oracle measurements. SHA-256 bindings for the archived artifacts are:

| Artifact | SHA-256 |
|---|---|
| `hill_material_point_oracle.py` | `839592eb248a5202cfbe62e24e2f514a9806e18adda8921ff2380f5cc0199d7f` |
| `constitutive-path-audit.json` | `ff3839f16d99099b7845bd32e3e791b9b7d7df357e6d7b4ca2f438ed3240d1e2` |
