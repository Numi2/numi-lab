# Numi Human static reaction handoff

The native large-state stand owner accepts an explicit environment-major `generalizedForcePreload` stream. The source static solve supplies support, scalar equality, active position-limit, and registered passive-coordinate reactions through that stream. The native owner adds the stream to the MyoSim generalized force before its mass solve; gravity, activation dynamics, contact projection, and joint equality projection remain owned by the native step. Root assistance is not used.

The persistent probe also carries only the six witnesses with positive static normal reaction into the dynamic contact set. This keeps the unilateral problem identical at release instead of admitting all ten geometrically near-plane witnesses.

On the M4 Pro Mac mini with the canonical one-adult payload, a clean rebuild reproduces the 12.5 microsecond one-step release at `3.00897479057 m/s2` and `7.83274299465e-5 m/s` maximum velocity delta. The source static certificate remains `balanced=false` with normalized residual RMS `0.86077456182`, so this is a bounded reaction handoff rather than a solved standing state.

The original 64-step value of `5.75767946243 m/s2` is retained in the v1 receipt as historical evidence, but it is contradicted by a clean rebuild and deterministic repeat under the same commit and canonical payload hashes. The repeat measured `7.71253347397 m/s2`, `3.95996576117e-5` normal impulse, `0.00596422795206 m/s` maximum final velocity delta, and `3.80200799555e-6 m` maximum configuration delta over `0.8 ms`. The corrected result is recorded in `media/numi-human-static-preload-v1/qualification-rerun-20260914.json`.

This remains a bounded release and handoff receipt, not a 10–60 second assistance-free standing qualification. Perturbation recovery, walking, subject calibration, deformable materials, and blood-to-tissue closure remain open.
