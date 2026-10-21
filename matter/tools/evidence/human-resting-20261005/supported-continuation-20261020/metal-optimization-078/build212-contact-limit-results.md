# Contact limiter and native regression checkpoint

This increment adds a default-off collision-gap limiter and exact per-lane minimum witnesses. It does not qualify 0.1x performance or a sustained integrated Human run.

The retained failing witness is a kidney/spleen edge contact at 18.970 micrometres separation. The prior limiter subtracts the 10-micrometre barrier activation distance, restricting the correction outside the force-bearing barrier region. The opt-in `NM_FEM_DEFORMABLE_COLLISION_GAP=1` uses the existing 1.113-micrometre collision thickness for this gap bound. Existing CCD, barrier energy/Armijo, constitutive validity, true-residual and publication checks remain mandatory. Geometry and forces are not clipped.

At 31.25 microseconds per root with persistent vascular low words, the previous configuration fails after 11 accepted roots. Build212 with the new gap bound passes 12 roots and then 64 roots (2 ms simulated). All 64 recorded statuses are healthy; maximum final mechanical relative residual is 9.24583583e-6. The 64-root body interval takes 271.559879958 seconds: 0.0000073648585x. Process time including initialization is 285.32551975 seconds. This diagnostic run uses Metal validation and is not an optimized sustained-throughput measurement.

The 3-root 125-microsecond comparison against the earlier build207 candidate misses the 1% equivalence gate at maximum J (1.446%). The option therefore remains off by default. Stability improvement is separate from numerical equivalence and physiological qualification.

A separate trace bug copied environment-multiplied object definitions from shared object storage. The copy now uses objectCount, and every trace arena is bounds-checked before encoding. Missing telemetry buffer binding is also corrected; the allocated scratch is bound regardless of whether telemetry is displayed.

Native regression on build213 passes 77,438 FGMRES checks with Metal validation and the default 32-step, two-environment moving-cavity/replay/rollback fixture. The 31-test source contract passes after updating the four diagnostic fallback bindings. The measured binaries include explicitly disabled experimental cavity work; the archived source patches and manifests describe the complete tested closure, rather than attributing the run to a clean commit alone.

Experimental cavity precision passes its direct volume/JVP/force/work tests and the moving-wall/replay portion; the all-fixed-wall control still fails. The trace identifies a zero-flow upwind linearization selecting the wrong donor for a tiny reverse-flow correction, leading to a negative tracer direction in an empty compartment. This remains under repair, and the compensated cavity option remains off.

The fresh-RHS no-work skip also remains off: cache preservation and clearing both miss the 1% comparison gate. Force-full preflight is exact on all 17 endpoint metrics, separating the measurement from the skipped cache-production effect. Failed trials are retained in this archive.

All execution is SSH numi, Apple M6. No visual qualification was performed. See invocation JSON for exact assets/configuration and source manifests for revisions, hashes, toolchain and binary identities.
