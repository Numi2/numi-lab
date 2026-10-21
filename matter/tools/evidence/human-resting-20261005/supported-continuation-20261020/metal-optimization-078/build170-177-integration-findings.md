# Integrated optimization review: builds 170–177

Execution host: SSH Numi. No visual verification. These runs remain initialization mechanics probes, not sustained or settled integrated qualification. The 0.1x minimum is not achieved.

## Measured priorities

Whole-stage timing is emitted as `matter_gpu_stage` on stderr. Build172 samples 0.501402251 seconds assembly, 5.33987729 linear solve, and 0.350443291 globalization. The second of four mechanical microsteps consumes 3.958515295 seconds in linear work: all 24 Newton solves exhaust 64 FGMRES columns. Completed tails are not the whole problem. Three roots previously showed declining work, but only 250 microseconds of post-first-root timing; that is not resting throughput.

Stage-profile control perturbation differs by pair: build171 -0.16%, build172 +2.81%. Do not infer tiny gains from selected kernel samples or one pair.

## Accepted equivalent-algorithm probes

- Build170: 16-node / 48-DOF SIMD32 local patches versus 32-node patches: all 17 endpoint audits pass 1%; body 9.740490 to 8.937207 seconds. Single pair, not a qualified sustained gain.
- Build172: compensated parallel back-substitution retains the original initial ordered solve. All 17 audit values exact; 9.449620 to 9.371050 seconds. The broader build171 parallel initial solve failed the acceleration gate by 3.654% and is not promoted.
- Build171 actual coupled-mass owner: exact-candidate / implicit-body20 changes the layout from 28450 to 28578 slots. The borrowed path is observed and all 17 outputs exact. Body 11.868940 to 11.590140 seconds; a single pair under this distinct system, not a default-layout gain.

## Conditioning and model experiments

Build172 full vascular coarse space has 32 modes, including 24 connection flows and the cavity volume/pressure pair. Its factor was rejected and it applied zero corrections. Exact output agreement therefore does not establish useful preconditioning. It regressed body time from 9.485390 to 13.056680 seconds. Equilibrated local factorization is being checked while preserving rank rejection; no speedup is assumed.

Build171 anatomical region refinement did not fix reduced-model fidelity: reaction relative L1 5.2331%, shell weighted Green-strain relative L1 149.43%. High+low captured positions are used. More than 99% of shell tetrahedra exceed a conservative Float32 ULP strain bound by 10x, so quantization does not explain the discrepancy. Broad skin reduction remains unqualified. Optic-only coordinates/preconditioning preserve fine skin and are the next bounded comparison.

## API validation failures retained

- Build173: CPU-encoded indirect command buffers allocated in private storage; native validation aborts. Four CPU-recorded ICB allocations changed to shared storage in build174.
- Build174: zero-face cohesive residual dispatch binds absent force storage. Build175 returns before encoding the provably empty domain.
- Build175: zero-face cohesive operator dispatch binds absent operator storage. Build176 applies the equivalent guard to that additive-only operator.
- Build176: disabled contact trace binding has 384 bytes but the shader requires 1216. Build177 allocates a typed-size disabled diagnostic binding; trace slot controls remain disabled.

All aborted trials are failures, not accepted throughput evidence. Earlier ICB timings without API validation remain provisional. The 32-mode equilibrated coarse fixture passes 86 checks with Metal API validation, including an ill-scaled nonsymmetric solve and exact rank-defect rejection. Its integrated effect remains to be measured.

## Still required

A useful full-model preconditioner, supported periodic resting checkpoint, accuracy-qualified larger timestep, three matched pairs, 10/60/330-second coupled runs, intervention/recovery, and >=0.1x end-to-end presentation qualification remain open. No physics gate has been relaxed to obtain speed.

## Build177 integrated API-validation outcome

After the four storage/binding corrections, the complete exact-candidate / implicit-body20 native probe completes one accepted 125-microsecond root under `MTL_DEBUG_LAYER=1`; the actual transaction source-mass ICB route is observed. Compact graph fixtures and the borrowed-owner fixture also pass under API validation. This is API execution qualification for the exercised short path, not physical steady-state qualification.

The equilibrated 32-mode coarse native pair also completes under API validation, but all **96 of 96** coarse factors are rejected and zero additions occur. Outputs are exact; body cost regresses **7.096689958 to 11.169896416 seconds**. This disproves scaling alone as a sufficient fix. First-active-matrix capture is needed to distinguish unsupported/zero/dependent coarse coordinates. No coarse optimization gain is claimed.
