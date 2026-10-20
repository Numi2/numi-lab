# Solver/material optimization qualification

All execution uses SSH Numi (Apple M6, 12 GPU cores, 16 GiB, macOS 27). No visual verification. Frozen binaries, Metal libraries, source patches, manifests, exact arguments, device and asset identities remain bound to each trial.

## Measured result

Build14's three matched four-root pairs pass the existing 1% numerical gate and terminal FEM/fascia checks. Median mechanics speedup is 1.644889783x against the same-build scalar evaluator control. The candidate simulates 0.5 ms in 17.83–17.90 s (approximately 0.000028x). These are release transients, not supported-rest or integrated viewer qualification. This does not meet 0.1x.

The independent tangent audit compares the production native specialization with the unspecialized expression interpreter at nine basis directions plus six seeded mixed directions per active element. Cooperative mode passes. The earlier serial complete-map mode failed with a zero cached response at source tetrahedron 3315 versus a nonzero interpreter reference; this failed trial is retained. Build16 writes each native column directly to the device cache after shared prelude evaluation; its independent interpreter audit and physical certificates pass. No compiler defect or aliasing cause is established.

Build15 material-specific pipelines pass independent tangent checks, but one warm four-root run takes 18.55 s versus 17.67 s without per-material pipelines. This option is not promoted as an improvement. A principal vascular preconditioner trial completes in 16.05 s but fails the 1% max-J comparison (1.13438 versus 1.11646); its physical certificates pass, but algorithm-equivalence qualification does not.

## Implemented paths and scope

- Distributed CGS2/reorthogonalization, Armijo norms, correction and true-residual reductions; object-range tiled geometry validation.
- Explicit active work domains, empty transport exclusion, active-corner incidence radix construction, parallel Morton sorting and hierarchical radix prefixes.
- Native material derivative bundles with shared expressions, consistent tangent selection and eligible shared local factorization across nine right-hand sides. The current Human's twenty materials are stateless: the fifteen-state factorization path is tested separately and cannot explain Human speedups.
- Cooperative stateless tangent construction; native per-material pipeline variants remain optional after a slower trial.
- Existing regional/coarse machinery extended with FEM tangent contributions and conservative basis projection. This is a preconditioner, not a reduced-coordinate physical model or exact coupled Schur condensation.
- GPU convergence-conditioned ICB subgraphs cover CGS2 columns and selected solver vectors/tails. They do not cover the complete Newton graph.
- Explicit initialization gravity continuation and supported-pose experiment controls; these have not established a settled checkpoint.
- Opt-in residual-adaptive forcing, vascular principal/cavity options, startup archives and asynchronous pipeline creation remain independently measurable experiments.

No physics tolerances, contact laws, material laws, state precision, rollback or physical acceptance gates are relaxed.

## Rejections and blockers

The 0.5 ms timestep trial fails final mixed certification at relative residual 0.0001003973 against the authored 0.0001 tolerance. This is not a geometry-alpha failure. A selector-6 coupled tangent trial rejects a vascular line search at right-ventricle row16; the same witness predates the new coarse preconditioner. These failures must be corrected rather than hidden with a larger budget or weaker certificate.

The regional Human preconditioner originally crashed because late immutable uploads reused a finished blit encoder. A separate fresh, explicitly finished UploadPlan fixes that ownership error. Build16 native initialization confirms the crash is fixed; the physical trial still rejects right-ventricle row16 and is not qualified.

The 2 ms explicit spring configuration fails its existing stiffness/inertial admission check. Implicit spring kernels remain fail-closed until the actual Brain/respiratory prepared transaction is connected. Current standing Brain publication is immediate; the existing prepared Brain API needs a versioned bridge. No substitute Brain or fabricated acknowledgement is used.

## Remaining qualification

Full coupled Schur condensation, actual hybrid reduced mechanical coordinates, accepted-step recycling for Human's selector6, complete conditional solver graphs, the Brain prepared bridge, supported resting convergence and timestep qualification remain incomplete. The 10/60/330-second coupled runs and end-to-end 0.1x requirement are not achieved. Short mechanics numbers must not be promoted to sustained physiological or native-presentation performance.

## Build16 additional evidence

The 0.25 ms initialization trial completes 32 accepted steps: 0.008 simulated seconds in 198.135778792 body-wall seconds, RTF 0.0000403763541787. Terminal FEM and fascia checks pass. Initial support force is zero and balanced=false; terminal support force is also zero. It remains a release transient, not a supported resting checkpoint or sustained performance claim. The 0.5 ms trial still fails final residual certification.

A warm four-root serial/cooperative pair takes 17.184185084 / 17.711585375 seconds and passes the existing numerical comparison. This single-pair difference is provisional, not a three-pair qualified gain.

Focused checks pass for transaction rollback, parallel Armijo, GPU projection, contact incidence, skin-spring linearization, materials and vascular cavities. The standalone FGMRES checker fails because its PSO creation omitted function constant 34; a source fixture fix awaits build17.

Stage-only profiling at root3 reports assembly 0.598937247 s, linear solve 2.732613005 s, globalization 0.862831831 s. Selected cache profiling records 96 cache spans totaling 0.480491371 s, but combined stage/kernel profiling drops empty-encoder stage timestamps. Detailed operator profiling also exceeds 4096 counter samples (22821 dropped). Neither censored output establishes whole-run kernel proportions. Build17 adds bounded capacity selection, invalid-record reporting and a nonempty profiling boundary marker. M6 supports stage counter sampling but not dispatch counter sampling.

Build16 active-column/cycle/Newton counters are useful, but its budget_exhausted_solves field is invalid: it compares against restart length rather than the total linear budget. Build17 moves that measurement after the final restart true-residual certificate and compares the actual total budget.

See build14-matched-results.json and each run's invocation/process/summary records. Failed and censored trials remain retained.


## Build17 confirmation

The rebuilt standalone FGMRES fixture, Armijo fixture and production transaction rollback all pass (3/3). The two-command Newton-tail ICB path passes terminal physical checks and the existing 1% numerical comparison against a same-build four-root control. Body wall time is 17.148825 versus 17.183202375 seconds (1.0020x); this difference is within noise and is not claimed as a performance improvement. It suppresses two proven inactive commands, not the complete nonlinear graph. Exhaustion telemetry is now measured after terminal true-residual processing using the configured total budget.

The build17 attempt to raise a single timestamp buffer to 65536 samples is unavailable on M6: the driver limits its byte length to 32768 (4096 timestamps). This trial does not provide usable operator timings and remains retained. Source work for build18 shards the bounded capture across resources; it is not included in build17's measured claims.
