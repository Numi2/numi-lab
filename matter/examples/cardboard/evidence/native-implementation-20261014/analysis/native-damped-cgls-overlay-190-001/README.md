# Optional damped full-zero CGLS pilot (source190)

This overlay adds a default-off, source-bound numerical experiment for the single-Newton CGLS fallback. It solves the damped least-squares objective `||A x - b||^2 + mu^2 ||x||^2` from `x=0`, with the pilot restricted to `mu=1e-3 kg`, CGLS budgets 32 or 64, no right scaling, no warm start, and no primary-policy mode. The damped direction is not covered by the original Newton certificate. The physical operator and the existing true-residual, nonlinear Armijo, acceptance, and rollback checks are not changed.

The patch is based on source188 `source-scaled-contact-geometry-188-buildfix002` and applies cleanly with `patch -p1`. It changes seven source files, listed in `validation-report.json`; the three helper files copied into `candidate/` are unchanged baseline dependencies and are not in the patch.

Validation: source-bound macmini build190-002 compiled `numi-matter-cardboard-probe`, `numi-matter-cardboard-coupon-resume-check`, and `numi-matter-cardboard-warm-start-check`; the coupon-resume CPU CTest passed; 13 environment-contract tests passed; and the independent nonsymmetric 2x2 spectral Tikhonov oracle passed. The warm-start checker was compiled only. No native GPU checker or coupon solve was run.

The first source-bound build attempt (build190-001) failed before CMake configuration because the noninteractive SSH PATH omitted `/opt/homebrew/bin`; that failure and the successful fresh retry are both retained. This is not a source compilation failure.

See `validation-report.json` for exact source, build-artifact, patch, and validation hashes. `validation/first-build-attempt-001.json` records the failed environment preflight.
