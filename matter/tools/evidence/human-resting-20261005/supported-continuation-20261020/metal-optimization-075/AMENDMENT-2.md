# Serial global-measurement correction

Before measuring this increment: replace whole-body serial Armijo norms with tiled partials and a compact reduction; replace object-by-all-tetrahedra validation with independently dispatched object-range tiles and deterministic failure finalization. Correct direct and cached-ICB CGS2 grids to match the shader's active domain. Extend remaining solver measurements using equivalent masks and mandatory certification.

Retain legacy kernels and opt-in control NM_FEM_DISTRIBUTED_GLOBAL_MEASUREMENTS=1. Validate native fixtures first, then four-root release runs against the same frozen ea2cbd84 control and the prior three-pair results. Preserve the existing max(1e-7, 1% * max(abs(a), abs(b))) comparison. No predicted speedup is claimed for these new kernels.

Use whole-stage telemetry and phase-rotated samples only as diagnostic observations, alongside uninstrumented timing. Initialize/compilation time remains separate. This release transient remains distinct from supported-rest and full-coupled end-to-end qualification.
