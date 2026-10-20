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

The qualified paths preserve physics tolerances, contact laws, material laws, state precision, rollback and physical acceptance gates. Default-off stopping and timestep experiments are reported separately below and are not promoted when their numerical comparison fails.

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

## Builds18–19 follow-up

Timestamp captures now use bounded shards of at most 4096 samples per resource on M6. Build18's full root3 operator capture has no invalid or dropped samples: assembly 0.600294 s, linear solve 2.979085 s, globalization 0.862148 s. FEM element application / node gathering / vascular application / cavity application total 0.416326 / 0.419280 / 0.213099 / 0.036886 s over 6144 spans each. This is observed instrumentation, not scaled sampling. Four-root body timing is 18.102456 s instrumented versus 17.136294 s uninstrumented (5.64% perturbation); do not combine these into an uninstrumented cost prediction.

The actual Human selector6 additive local FEM graph path builds 949 patches. Combined with principal vascular and cavity-Schur options it passes one accepted root and the existing 1% comparison, but takes 22.114833 s versus 4.034879 s for the same-build ordinary control. This is a slower experiment, not an optimization gain or exact coupled Schur condensation. The synthetic retained-topology graph fixture independently passes zero RHS, completed tails, additive correction, invalid and singular factors, and bitwise serial/parallel factor parity. Its spring operator is explicitly synthetic.

Build19 tests domain-separated nonlinear stopping, using the independently authored mechanical and vascular thresholds while retaining final certificates. It passes four-root physical checks in 9.917686 s versus 17.007002 s (1.7148x), but fails the existing 1% equivalence gate: maximum J differs by 1.4% and peak attachment reaction by 6.9%. It stays default-off and is not a qualified gain.

Existing admitted passive-FEM rate caps are also tested. Cap0 completes four roots in 8.405019 s versus 17.007002 s, but fails determinant, displacement, attachment and body-contact comparisons. This is not a qualified timestep change. Cap1 results and witnesses are retained separately. Reducing scheduling work without numerical agreement is insufficient.

The 0.5 ms root-step trial with explicit gravity continuation accepts 14 roots (7 ms), then rejects geometry at object19/element4214. The prior four-root invocation was invalid because its horizon was shorter than the declared ramp; it is retained as preparation failure, not a simulation failure. No supported checkpoint or larger timestep is qualified by these trials.


## Build20 recycling integration

The accepted-step FEM Z0 direction is staged per active Newton update and promoted only after the Human owner status, environment, completion count and same-command-buffer lease pass. Completed tails preserve the last active correction. Constraint/contact/timestep changes invalidate reuse; only a direction is stored and its action is recomputed by the current full operator. Owner rejection changes solver scratch only, never published physical state.

The first build failed checked C++ narrowing of gate offsets; the repair uses explicit checked uint32 casts. Review also caught that the nonzero flags value 5 was incorrectly classified as malformed, which would have prevented all reuse. Both failures are retained. A repaired native fixture passes owner rejection, completed tails, zero fallback, contact/dt invalidation, basis injection and changed-current-operator tests; its first PSO attempt failed because the new FEM operator fixture lacked explicit function constants. That fixture specialization was repaired before testing, without altering production equations.

Two initial Human invocations fail their existing selector admission at initialization. The admitted coupled selector6/coarse/principal/cavity configuration completes one root in 9.629004 body seconds, but uses all 6144 encoded columns and all 96 Newton attempts. It is not a measured recycling gain: a prior accepted direction does not exist during the first root. Actual Human use telemetry and multi-root comparison remain necessary.

During integration a stale whole-file agent patch removed the then-unbuilt graph-lag and cache-ICB edits. Those edits are being restored from isolated narrow patches; build20 cannot establish their behavior. No stale source snapshot is substituted into the retained build20 evidence.

## Build21 combined validation

Eight focused tests pass: production rollback, regional graph completed-tail policy, cohesive Armijo tiles, CGLS arena planning, native FGMRES true-residual/recycling, capsule Armijo, full-contact Armijo and distributed Armijo measurements. The repaired recycling fixture verifies actual injection count as well as the current-operator action; no stale operator result is accepted.

The expanded Newton-tail ICB conditionally includes force assembly, tangent-cache preparation/build and residual gathering for the eligible ordinary cache path. Unsupported cooperative/specialized/projection/forced-rebuild paths retain direct scheduling. Mandatory residual/final/physical processing remains outside. A matched four-root pair passes all 17 numerical metrics and physical terminal checks, but has no measured gain: 17.096627 versus 17.098094 body seconds (0.999914x). This is working conditional scheduling, not a performance claim.

Actual accepted-step recycling executes in the Human selector6 path: its counter reaches 126 injections before a geometry rejection at root2 (object20/index15687). The matched configuration without recycling also rejects, at root1 on a mixed residual of 1.02259e-5 versus 1e-5. These different failures do not establish that either path is equivalent or qualified, and no recycling speedup is reported. The previous first-root run could not exercise accepted-step reuse.

Lagged local graph factors complete one root in 11.920384 body seconds versus 19.296113 without lag (1.61875x). However maximum attachment reaction differs by 1.40%, exceeding the existing 1% gate. It remains default-off. The graph configuration is slower than the ordinary solver; this result is not a global Human gain.

The repository comparison entrypoint is matter/tools/human_performance_compare.py. It consumes the retained summary files, enforces finite values, exact scalar/vector shapes, required metric coverage, successful terminal checks and matched simulated durations before reporting a mechanics speedup. Inline checks cover omitted metrics, truncated vectors, NaN, booleans, owner failure and mismatched durations. This avoids silently passing zip-truncated vectors or failed runs.

Build21 does not complete exact Schur condensation, actual hybrid Bq runtime integration, a full operator ICB, a supported checkpoint, or the 0.1x end-to-end target. Those remain implementation work. All measurements here are short release-transient mechanics checks with presentation qualification pending.

## Build22 conservative domain-stopping qualification

A new opt-in internal exponent tightens only the mechanical domain convergence gate: NM_FEM_DOMAIN_MECHANICAL_TIGHTENING=3 applies 2^-3 to the authored mechanical threshold (1.25e-5 when the authored absolute threshold is 1e-4). The vascular bound, true-residual processing and all final physical tolerances remain unchanged. The legacy path is unchanged when the domain latch is off. Native tests cover passing/failing tightened domains and unchanged legacy full-norm behavior; three focused checks pass.

Three four-root matched pairs all pass the 17-metric 1% comparison and terminal FEM/fascia checks. Control/candidate body seconds are 17.083606833/11.647453834, 17.288182834/11.655798959, and 16.977690083/11.711979084. Median speedup is 1.46672458x. Candidate RTF is 0.00004269–0.00004293x for 0.5 ms simulated per trial. This is a new short-mechanics improvement against a same-build native control, not an integrated 0.1x result, a supported checkpoint, or a multiplied forecast from prior gains.

The weaker exponent2 trial takes 9.938724 s but repeats the determinant/attachment-force comparison failures; it is not qualified. Combining accepted exponent3 with Eisenstat-Walker takes less time but fails attachment-force parity (3.26% relative to the same-build control). Keep that combination experimental.

An exhaustive operator/stage capture of the qualified exponent3 path has no invalid/dropped records. At root3 it observes assembly 0.562198 s, linear solve 1.822354 s and globalization 0.860927 s, including 6144 spans of each profiled operator. Body time is 12.674065 s versus median uninstrumented 11.655799 s (8.74% perturbation). These are exact observed spans, not scaled phase samples. Inactive graph scheduling remains substantial; further complete operator/Newton subgraphs are still unfinished.


## Builds23–25 complete operator graph and Schur primitives

The regional source/interface classifier and parallel A_ii extraction are implemented and pass synthetic CPU/GPU checks. They are prerequisites only: no integrated Schur-condensed global solve is claimed. The initial build exposed a shader-local scope error, an unused C++ capture and invalid host vector iteration; these were repaired. Two initial fixtures then failed because a supposedly covered test topology had orphan nodes and the matrix expectation used an incorrect component map. Repaired checks pass; failed logs/source cohort23 remain retained.

The optional full Arnoldi operator path now records a FEM/field prefix and a rigid/contact/support/vascular/constraint tail, leaving the existing borrowed rigid mass callback between them. GPU convergence and status choose zero or complete execution ranges; true-residual and physical acceptance processing remain mandatory. The initial Human trial crashed in AGX setComputePipelineState because seventeen operator PSOs lacked indirect-command support. Build25 explicitly enables that descriptor property for the exact command set. The crash report and failed cohort24 remain retained.

Build25 completes four Human roots and passes all seventeen numerical comparisons plus terminal FEM/fascia checks against the same-build domain-tight3 control. Body wall time is 11.709580 s versus 11.631817 s (0.993359x); there is no measured performance gain. This first graph implementation reconstructs constants/graphs at the first restart of each Newton solve. Reusable immutable graphs with GPU-updated constants are still being implemented; no gain from that follow-up is claimed.

The 0.5 ms root-step trial with domain-tight3 accepts only three roots before mixed residual certification rejects object1 (residual 0.000105324 against 0.0001). It does not qualify a larger timestep or supported rest. The 0.1x minimum and long integrated qualification remain unmet.
