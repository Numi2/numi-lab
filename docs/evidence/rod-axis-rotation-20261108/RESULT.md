# Axial rod orientation and current-runtime comparison

Tested native Apple GPU with integration checkout 5bebfde19, but the executed library and GPU shader hashes in executed.sha256, not the checkout name, identify the actual pre-existing integrated build. The candidate shader is the unchanged research version at 930db6091, linked with current integrated shader objects and runtime. The parent of the original research shader is byte-identical to the current integrated DiscreteElasticRod.metal source. No production shader or contact option was changed.

| Arm | x-axis symmetric / asymmetric / moving | y-axis symmetric / asymmetric / moving |
|---|---|---|
| Integrated production | incorrect acceptance / rejected / rejected | incorrect acceptance / rejected / rejected |
| Direct axial research candidate | pass / pass / pass | rejected / rejected / rejected |

For each production symmetric case the published middle node moved 0.437637 mm and 0.437636 m/s despite a stationary analytic reference; the support forces were about 0.000622 and -0.000657 N versus +0.089379 and -0.089379 N. The research candidate passed the same x-axis state and force gates; the worst x-axis force error is below 1.6e-7 N. For the y-axis, the research candidate rejects all three at status 4. Rejection contains the original false success in that orientation but is not useful-motion completion. All rejected outputs and codes are retained. This is one straight, zero-gravity, three-node unloaded-contact fixture, not evidence of general sutures, material realism, bending, friction, device-resident integration, large-offset robustness or complete state restoration.

The direct solve checks and operates on world x components. Unsupported orientations fall through to its separate stretch certificate; this outcome is consistent with that admission boundary, not a rotation-invariant repair. The next research change must address physical orientation without silently admitting bent, off-axis-velocity or other unsupported geometry. It needs independent state, support-force and work references, arbitrary rotations, large-offset precision tests, multi-step restoration and coupled-owner qualification before any promotion.

The first probe generator had a Python quoting syntax error and never generated or executed a probe. Its source is retained as make_probe.failed.py; fix_probe.py reproduces its correction. PLAN.md, hashes, source, native build commands, outputs and results are retained. The baseline x-axis output is identical to the previously retained integrated result. Results do not include timing claims.
