# Paired mixed-contact normal correction: retained failure

**Decision: do not promote or tune this candidate.** The isolated normal-pair correction gets past the earlier rejected state but does not complete the predeclared sustained fixture. Neither the directional CCD nor augmented-Lagrangian contact experiment was enabled. No production shader was changed.

The test uses the retained 20 x 1 ms sphere, plane and three-node rod fixture, with friction enabled and CCD disabled, the same bounded research split solver, a two-sweep minimum and a 128-sweep maximum. The fresh unmodified baseline reproduces all previously retained common status, state, sweep, residual and contact-row fields.

| Result | Baseline | Paired candidate |
| --- | ---: | ---: |
| Accepted steps | 7/20 | 12/20 |
| First rejection, zero-based step | 7 | 12 |
| First rejected normal residual | 8.64509e-5 m/s | 4.18867e-5 m/s |
| First rejected tangent diagnostic | 1.25564e-4 m/s | 0.00260291 m/s |
| Accepted physical time | 0.007 s | 0.012 s |

Both candidates reject with status 11 and restore exposed q/v and complete rod-node records to the previous accepted step. Capture-free versus captured exposed final state and repeated captured traces match. These checks do not establish restoration of hidden contact caches, scene-body state at every failed step, or all owner state. Timings include capture and three internal runs and are not a performance benchmark.

On the accepted 12 ms candidate prefix, the maximum 1 ms ground normal-force difference against the fixed-128 *discrete* control is 0.1242903% of that control rollout peak; the integrated ground normal-impulse difference for the same prefix is 0.0004450%. The required full-duration gate is **not met** because the candidate rejects eight steps. The fixed-128 control is not a force-qualified physical reference. The maximum separately summed momentum-balance component over the accepted candidate prefix is 5.80e-9 N s. Estimated ground normal and tangent contact work over that prefix are +0.00065868 J and -0.00124800 J respectively. These estimates omit internal rod-contact work and stored energy; they are not a complete energy balance.

The independent CPU FP32 formula check agrees with an FP64 active-set enumeration for all four activity patterns and the captured failing matrix; it rejects the selected singular, indefinite, negative-diagonal and nonfinite examples. This checks the arithmetic independently, not execution of the Metal code. At the captured state the solution activates only the second rod row, unlike the inadmissible unconstrained solution. The native candidate uses an implicit factorized rod cross response and retains the original row relaxation, tangential update and stopping rules. Those stopping rules have previously documented small-impulse limitations and are not a physical complementarity certificate.

The research implementation does not certify every promised fallback: it does not explicitly validate all cone fields before entering the block, and the final two projected row writes are checked individually rather than published atomically. The tested fixture produced finite values, but malformed or nonfinite cone cases were not qualified. This is an additional reason to leave it isolated. The unused poses, zero-friction controls, independently reconstructed tangent response, and full transactional restoration were not run after the primary completion gate failed. No general solver, CCD, friction, surgery, or performance claim follows.

Reproduction on ssh numi: run active_set_check.py with the retained diagnostic venv, run build.py to generate the isolated Metal shader, then execute the captured probe and commands recorded in commands.txt. analyze.py compares the raw trial with the retained fixed-128 and earlier baseline traces and includes per-step momentum, normal work, tangent work and impulses. The generator depends on the retained baseline research source on the host; binaries and AIR files stay on the host, with identities in inputs.sha256 and runtime-sha256.txt. The next useful investigation is the first rejected candidate tangential equation and the coupled normal-tangent operator, alongside independent physical force qualification and complete owner restoration. The legacy thin-impact CCD completion and the FEM-plane force-qualified control remain open; do not reopen performance tuning of the disabled experiments.
