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

## Build26–27 hybrid mapping and reusable operator graphs

Build26 introduced an opt-in hybrid coordinate experiment that retains full material/contact evaluation and uses x=Pq with adjoint P^T projection. A focused GPU test first exposed a fixture basis mismatch (one-hot anchor modes were outside the regional sparse map), then a genuine lift defect that zeroed unsupported fine-complement rows. Build27 uses a regional-only hybrid fixture while retaining separate full-basis anchor checks, and leaves unsupported rows unchanged. Four focused tests and eleven regressions, including production transaction rollback, true-residual, Armijo, projection and Schur checks, pass on M6.

The integrated hybrid attempt stops at admission before advancing physics. Its current certification action does not yet include all vascular/support/constraint terms. This gate remains closed; the coordinate experiment is not an admitted Human reduction, and no reduction speedup is claimed. Its current storage is padded to existing full-layout capacity.

The full Arnoldi operator ICB now retains per-column graphs while refreshing their constants on GPU before each invocation; shape changes invalidate the cache. The initial build failed an unnecessary __bridge cast of an existing raw pointer; the cast is repaired. A matched four-root run passes all 17 numerical metrics and physical terminal checks, but takes 11.820848333 body seconds versus 11.674079208 for the control (0.987583875x), so this is not a measured gain.

Residual-adaptive forcing with additional tightening exponents 1 and 2 also completes and passes physical terminal checks. Exponent1 takes 11.388096625 seconds, but max-J and anchor reaction differ by 1.778% and 2.267%; exponent2 takes 11.885897750 seconds and differs by 1.778% and 1.461%. Both fail the existing 1% numerical comparison and remain experimental.

The exact Human Schur partition reports 949 patches, zero eligible patches, zero interior patch nodes, 29,671 interface patch slots and 14,202 externally anchored nodes. This partition cannot reduce the global solve. The same graph trial rejects before accepting a root (code9); neither its physical state nor condensation is qualified. The anchor reasons and complete coupled operator action are being corrected before further admission.

No supported-rest, larger timestep, sustained physiology, viewer or 0.1x qualification is established by these short release-transient trials.

## Builds28-32 complete operator and reduced-coordinate experiments

The optional hybrid now uses the same complete direct operator owner as ordinary Arnoldi, including rigid mass, active contact, support/constraint and vascular/cavity terms. Certificate operator state is copied privately so convergence latches cannot suppress its action. Mandatory certification and restart replacement remain enabled. The selected reduced regions remove 20,616 active mechanical coordinates (41,796 to 21,180 overall, 1.973x); full material/contact evaluation remains. This is not a whole-body 56x reduction.

Focused native/CPU tests cover hybrid tail rows, projection/preconditioner mapping, invariant layout, gravity-scale transactions and regional partitioning. The first actual reduced Human trials advance zero roots and fail vascular globalization; 0.1x and reduced-model accuracy remain unqualified. The tangent-block and mass-preconditioned variants both fail; logs are retained. No invented speedup is attributed to them.

The Schur classifier no longer labels every two-way-coupled node as an external anchor. Static surface maps now anchor their three emitted triangle vertices rather than all vertices of both adjacent tetrahedra; disabled faces remain excluded. This corrects partition metadata without deleting global contact or material equations. Actual global interface condensation is still unfinished.

Initialization now has an explicit transactional GPU gravity-load continuation through the existing owner. Default gravity remains exactly one. A 33-root/4ms ramp accepts 27 roots then fails vascular line search; the authored-tolerance variant accepts 25. These are preparation failures, not a supported checkpoint. At the rejected direction the actual trial agrees with the local tangent prediction to about 3.55e-11 while the correction increases the failing vascular row. Strict linear/coupled certification is being tested against that witness; no Armijo tolerance is relaxed.

Build32 default four-root comparison passes all 17 metrics and physical terminal checks against build30: 11.697530834 vs 11.679572625 body seconds (0.9984648x). Five focused checks and the regional Schur CPU fixture pass. Full-coupled true-residual certification also completes four roots (14.0185305 body seconds). Its 384 trace cycles differ from Arnoldi estimates by at most 6.357e-10 absolute, so broad false convergence is not established. The first diagnostic used an ignored strict-inner environment spelling; strict forcing is not claimed for that run. Build32 trace scope text retains the legacy label although the new bit8 path executes all coupled rows; a follow-up corrects the label.

Storage cleanup SHA256-verified 107 duplicate finished-run mrvpack files before atomically replacing them with APFS clones. It preserved paths and bytes, recovering roughly 29.6GiB. No sole source or evidence input was removed.

The actual post-fix Human Schur partition has 34 eligible patches and 48 interior patch nodes against 29,623 interface slots (949 patches; 10,299 external anchors, including 10,083 potential surface vertices and 216 cavity nodes). Condensing this particular partition removes only about 0.16% of patch slots; it cannot provide the required scale of improvement. Its native one-root trial also rejects before acceptance. A large local-condensation runtime expansion is therefore not justified by these counts alone.

Strict inner forcing plus full-coupled certification advances gravity continuation from the prior 27 accepted roots to 28, then fails vascular line search at row24. The last tested direction has residual -1.10351566e-5, Jd -1.5511614e-6, alpha0.00048828125 and trial-minus-linear prediction7.576e-10. Whole process134.5502seconds. This is a small robustness change, not successful initialization or a throughput improvement.

## Build33 mass-normalized hybrid trial

Optional compiler-side regional mass normalization factors B^T M B once per region and solves triangular systems to export B L^-T; it does not explicitly invert the mass matrix or change the regional span. Sensitive anchors and source identities remain fixed. Six focused CPU/Metal checks pass, including ordinary full-coupled bit8 certificate tails with poisoned inactive padding, malformed offsets/capacity, transactional nonmutation, and mass/virtual-work/span tests for six- and twelve-mode bases.

The first normalized hybrid attempt still fails vascular globalization before an accepted root. Adding the existing principal vascular and cavity-Schur preconditioners permits one accepted root and full physical terminal checks: 0.125ms simulated in 10.165971292 body seconds. A four-root continuation accepts two roots then fails contact certification: object20, witness15626, diagnostics[16657,16804,0,4]. This is the first admitted reduced Human progression, but is neither sustained qualification nor faster than the ordinary path. No 1% force/5% deformation model-fidelity qualification or 0.1x claim is made.

Build33 ordinary four-root control passes all17 same-build numerical comparisons and terminal physical checks against build32 (1.0028468x within noise). Ten regression checks pass, including capsule contact Armijo, true-residual kernels, projection, gravity transactions and graph completed tails. Archived source patches are gzip-wrapped without changing their decompressed bytes or declared SHA256; whitespace-only cleanup after the build does not change executable semantics. The current default path is preserved.

## Existing vascular principal preconditioner with full certification

With strict inner forcing and complete coupled residual certification unchanged, enabling the existing exact vascular principal-block preconditioner completes the 33-step gravity ramp: 4.125ms simulated in141.7655515 body seconds (2.90973382e-5x), process153.3321seconds. Earlier strict-only continuation fails after28 accepted roots. This is a completed load ramp, not supported periodic rest.

For matched roots0..27, Krylov columns decrease56803→40853 (28.1%), active Newton begins1597→1414, exhausted solves372→358. Counter reductions are not wall-speedup measurements; nonlinear trajectories differ. The separate four-root release comparison passes all17 metrics and physical terminal checks against the same-build default but is slower (0.832648x), so the combined strict/preconditioned path is not promoted as a throughput gain.

## Builds34–36 measured conditioning and work-removal experiments

A six-mode global correction now constructs its RAP using six applications of the complete current coupled operator and the existing pivoted small LU. It remains opt-in and does not change coordinates. The initial same-build local/global one-root pair takes 9.567835125/11.184421625 body seconds, fails the existing 1% comparison (min-J 10.741%, max-J 1.833%), and is not a gain. Combining it with the existing regional graph reaches one root with 197 active Krylov columns, 14 active Newton begins and 15 measured nonzero global corrections; final physical FEM/fascia checks pass. Its 7.414779875 body seconds are not a matched throughput result. Global factor invalidity on the final completed tail is reported separately from cumulative actual uses.

The empty-field preconditioner replacement omits three proven empty field dispatches per encoded column while preserving mechanical pressure. Poisoned inactive rows, completed tails, accumulation and pressure tests pass in the native 51,462-check true-residual suite. A four-root matched pair passes all 17 metrics and physical terminal checks, but takes 11.622591667/11.642536667 body seconds (0.998286885x); no gain is established.

Completed-Newton backtracking can now collapse repeated trial scans only after proving a convergence latch and finite exact-zero xyz, pressure and temperature corrections. Trial-zero checks, rejection rules and the exact final bounded-fallback alpha remain. The native fixture checks first acceptance, invalid/nonzero/unlatched controls and 58 skipped tetrahedron trials. The first build exposed a fixture constant scope error; subsequent fixture runs exposed missing FC34 specialization and an undersized mixed-temperature correction arena. Those fixture defects are repaired and failed logs retained. Ten relinked regression checks pass. The actual Human matched pair passes all 17 metrics and physical checks with 11.685198334/11.602314 body seconds (1.007143776x); this difference is too small to claim a gain without repetition and actual skipped-work telemetry.

The zero-dynamic-contact census retains real attachments/cavities/field anchors and does not alter the physical partition. The existing overlapping graph has 949 patches, 251 eligible patches, 1,184 interior nodes and 28,487 interface slots even with all potential dynamic surface anchors removed. There are 216 genuine external anchors. This identifies the overlapping patch construction as a further condensation limit; it does not disprove condensation with different disjoint domains.

Actual source-bound endpoint capture now compares all 14,202 nodes and 54,133 tetrahedra. The current normalized skin/fat hybrid fails the engineering fidelity gates: shell volume-weighted strain relative-L1 error is 1.0233 (its reference deformation is very small), and the unreduced left optic nerve has a maximum absolute Green-strain error of 3.6488. Cavity fixed-reaction error is 1.4098%, and left ACL reaction error is 26.79%. Whole-volume strain averaging (0.2843%) conceals these local errors and is not an acceptance substitute. Force capture honestly excludes free-node internal force and per-node articulated attachment force; full physical-force qualification is incomplete. The comparator's twelve CPU cases pass. No hybrid model accuracy or speedup is claimed.

The repeated contact failure remains object20 (registered left optic nerve), witness15626, diagnostics[16657,16804,0,4], after two accepted roots. General diagnostics are disabled by the actual physiology owner; the first failure-only option covered Armijo narrowphase, while this terminal failure occurs in ordinary contact refresh. A follow-up enables the existing failure capture at that actual owner, without enabling blood/MPM or changing contact acceptance.


## Builds37–39: combined costs, exact contact witness and supported initialization

The combined regional graph/global-six-mode correction accepts four release roots but takes 31.452528042 body seconds (1.58969741e-5x). Reusing regional local factors within each physical microstep lowers that to 22.166831333 seconds (2.25562245e-5x). The latter passes all17 numerical/physical comparisons against the ordinary 11.685198334-second control, but is still slower (0.5271479x). Fewer Krylov columns do not establish a throughput gain when setup and operator applications grow. Neither option is promoted as a performance default.

Build38's ordinary four-root control completes in 11.660814625 body seconds (4.28786530e-5x), passes all17 comparisons and terminal physical checks against build36, and confirms default-path preservation. Its 1.00209x timing difference is noise-scale, not a claimed optimization.

Failure-only contact capture now reaches the ordinary contact-refresh owner and the ordered snapshot readback, independently of the general diagnostics disabled by the physiological owner. The reduced trial reproduces the same left-optic-nerve failure after two accepted roots. Exact Float32 start/end triangle data replay in the native checker: production and the old interval fallback return uncertified, not a confirmed collision. The generic parser rejects concatenated duplicate records; the retained first-record extraction records its parent hash and line offset.

The engine's existing offline support-pose fit now supports bounded root rotations through its actual exponential quaternion integration convention, including a new three-point tilted-support CPU fixture. Existing fit/recruitment/infeasible-output tests pass. Strict full-gravity initialization remains unqualified: build38 exposed a construction-pose admission guard, and build39 progresses through common-field t0 initialization before rejecting a later strict-owner/source-pose/callback guard. Both failures are retained; no supported equilibrium or larger timestep is claimed.

Seventeen completed duplicate run outputs were SHA256-verified and replaced by byte-identical APFS clones, recovering 5,050,355,712 bytes. Sources, current evidence, recovery paths and file bytes remain intact.

## Builds40–43 integrated consumers and exposed limits

The existing default full-body path remains numerically unchanged: build42 four-root release mechanics passes all 17 numerical comparisons against build40 plus FEM/fascia terminal checks. Body time is 11.641917458 s for 0.000500000024 s simulated (0.0000429483x); the 1.00528x difference is noise, not a claimed improvement. Presentation and supported periodic rest remain unqualified.

The captured optic-nerve CCD replay now certifies separation with a conservative interval separating-plane test inside the existing bounded fallback. The production and fallback tests report TOI1 and gap2.77072104e-6m; legacy/budget16 controls remain uncertified. No collision is accepted merely because the subdivision budget is exhausted. The existing optional completed-backtracking counter is now cumulative across the encode request. The actual four-root run reports zero skipped trials on every root and takes11.647385125s, passing all17 comparisons but offering no gain.

The actual Schur consumer now implements condensed RHS, interface operator action, interior recovery, and mandatory unchanged full-operator certification. A native nonsymmetric pivoting fixture passes as part of51,790 true-residual kernel checks. Initial Human admission exceeded the unchanged96MiB local-factor budget; deterministic smaller-core repartition now admits694patches with399interior patches and1,093eliminable nodes (10,299external anchors). The native trial then rejects before a Newton solve with descriptor diagnostics[16,3,702,1]. Investigation identifies a factor-descriptor buffer incorrectly bound as an extraction-descriptor buffer; the repair is pending. This path is not qualified or promoted as a speedup.

Skin/fat and both passive optic nerves now have an actual combined source-bound Bq consumer, mass-normalized regional modes, fine boundary complement, and adjoint force transfer. Its structural/rank fixtures pass. Build41 matched full/hybrid endpoints fail force and local strain gates; anatomy and engineering fidelity are not inferred from successful stepping. Local refinement adds three regions (468to504q) around measured witnesses without changing donor IDs or source ancestry. Build43 hybrid advances one root, but its matched full reference rejects optic contact before accepting a root, so no passing paired fidelity result is available.

Strict supported preparation now fits actual registered-skin points and recaptures the changing lowest features. A bounded recapture loop obtains three closed, noncollinear skin contacts on the unchanged flat support plane. The pelvis/foot and shoulder/pelvis trial poses both fail the original force-balance certificate (maximum acceleration2.0283 and1.62246m/s2 respectively). These are initialization failures, not resting checkpoints; no residual tolerance or hidden assistance was added.

The residual-Armijo later-trial dispatch experiment is inactive in this Human configuration and correctly rejects admission without residual-Armijo scratch. Its focused fixture required two setup repairs after scratch-poison probes; build43 passes the resulting test. The Human actually uses generic geometric backtracking; tiled work for that active path is being implemented separately.

A2ms release-step trial stops at the explicit-skin-attachment inertial bound before physics. All20skin springs remain explicit because the current resting transaction owner lacks the existing exact candidate-query bridge required for implicit coupling. The bound is preserved. Routing the existing candidate callback, rather than merely changing the spring count, is required.

Build43 passes12selected CPU/Metal regressions, including rollback/gravity, Schur sizing, true-residual contracts, projection/basis, capsule contact, and tiled Armijo measurements. Its full four-root tangent-cache profiling run takes11.639357583s. Root0 has96cache-column samples totaling35.939874ms; assembly543.278291ms, linear417.685046ms, globalization661.210288ms. These nested intervals overlap and must not be added across levels. The old8.06s cache bottleneck is not the current measured cost. Three phase-rotated whole-kernel profiles are being collected to identify the remaining active bottlenecks.

The minimum0.1x end-to-end requirement, supported checkpoint, timestep qualification, and long native-viewer qualification remain unmet.


## Build44 active-prefix contact compaction and Schur descriptor repair

Contact compaction previously used one SIMD group per environment to scan and clear all433,064allocated records even when the candidate prefix was about21,000records. The opt-in active-prefix path now clears unused records in parallel and performs the same stable compaction only over the candidate prefix. Exact contact bytes, active counts, active order and untouched index tails match the old path in a native two-environment fixture covering empty, small, full, over-capacity and shrunken candidate counts. The full native incidence/surface suite passes1,128checks.

Three matched four-root release-transient pairs all pass17numerical comparison metrics plus FEM/fascia terminal checks. Body times are control11.632577083/11.633075084/11.667698625s versus candidate11.013488/10.996585083/10.99851075s. Median throughput gain is1.057880696x. Candidate mechanics RTF is0.00004540–0.00004547x; this remains roughly2,200x below0.1x and excludes sustained supported physiology/presentation qualification. Alternating pair order retains startup/process timing separately. No claimed integrated real-time result.

The Schur descriptor fix separates extraction descriptors from LU factor descriptors and passes51,810native true-residual fixture checks, including a negative descriptor-shape case. The first native invocation omitted the required full-certificate opt-in and is retained as an admission failure. With certification enabled, actual condensation reaches local factorization and fails at patch326: pivot2.2988553769e-11 is below the unchanged3.4934732190e-11floor. This is local LU conditioning/admission failure, not a demonstrated final certificate mismatch. It accepts zero roots; eight used Krylov columns do not establish a performance result. Exact equilibration and original-matrix residual checks are being implemented.

All three phase-rotated build43profiles finish successfully. Their raw samples include an8.425209ms generic backtracking call and approximately2.986–2.987ms contact compaction calls. Samples are retained without multiplying them into unsupported whole-run estimates. The generic backtracking tiled replacement remains a separate next cohort. The actual20-material/54,133tet census reports zero material state variables throughout: no Human tetrahedron activates the15-state shared-factorization path. Optimizing that dormant family cannot accelerate this scene.


## Build45 tiled generic FEM backtracking

The actual Human generic selector now has an opt-in tiled geometry path. Trial0 evaluates each object's own tet ranges across independent256-threadgroups; ordered per-object reduction retains the earliest admissible alpha and exact minimum determinant. A GPU-generated later-trial range is zero when trial0 has already accepted; mixed-field objects retain their original merit selector. Legacy residual Armijo is a different, inactive Human branch and is not credited for this gain.

Three matched pairs with active-prefix compaction held on all pass17numerical metrics and physical terminal checks. Controls10.993008666/11.026997125/10.9990415s versus tiled8.224177625/8.143194333/8.144887584s give median1.350422751x throughput. RTF0.00006080–0.00006140x is still short mechanics throughput, roughly1,630x below0.1x, not a qualified integrated run. Do not multiply this by hypothetical timestep or reduction gains.

The first enlarged native fixture fails a stale assertion expecting58skipped tet-trials after its topology was enlarged to521tets. The narrow correction derives both fixture size and expectation from one shared constant; the failing log is retained. Shader behavior is checked separately against exact legacy alpha/J/trial/status, fixed nodes, the last partial tile, displacement-gradient reconstruction, exhausted rejection and mixed fallback in the next build.

The cache-ICB zero counter has a concrete policy cause: Human selects direct-fallback cache policy2, while its ICB admission required policy1. The next cohort admits exactly policies1and2 using the same builder/finalizer and unchanged invalid-cache direct fallback, with all existing material-pipeline/projection/forced-rebuild exclusions intact.

Disk accounting reached12GiB free after retaining the paired runs. Completed output files with identical full SHA-256 were replaced atomically by independently writable APFS copy-on-write clones; every byte and evidence path is retained.334files representing49,814,233,441logical bytes were verified, restoring available space from12,578,200to25,222,460KiB (about12.1GiB actually recovered). The exact manifest and before/after df are retained; no sole source, run trace or accepted evidence was deleted.


## Build46 validated cache ICB and publication boundary

The enlarged native backtracking fixture now passes in both reference-coordinate modes after correcting its stale expected count.12selected CPU/Metal regressions pass. The exact active-prefix contact path is now the default, with an explicit0override retaining the old control.

The policy2cache ICB executes4graph commands including2tangent-cache commands (previously2and0). Its four-root body time8.170729042s and all17comparison metrics pass against build45's matched8.224177625s trial. This one noisy0.65% difference is not claimed as a performance gain. First startup took69.58s due to new indirect-compatible PSO compilation; startup is reported separately and retained. The complete operator-ICB run also passes terminal checks at8.198799042s; it is not faster.

Existing host profiling, without per-kernel encoder splits, records command GPU times1224.55/1736.73/1701.40/2234.15ms for the four roots. GPU command spans remain most of the8.196610625s body run. Full-state readback occurs at the terminal root only. Physical time remains0.5ms, so no sustained physiology or0.1x qualification follows.

This publication freezes build46source. Exact Schur equilibration and further candidate-owner/conditional-prefix changes are subsequent unqualified work and are not included in its tested source blobs.


## Builds47–49: exact local equilibration, source ancestry and rejected approximations

Schur interior LU now factors Dr*Aii*Dc and solves with the retained permutation and row/column scales, with residual refinement evaluated on the original Aii. No explicit inverse or lowered pivot floor is used. Expanded pivot workspace is checked by CPU layout tests. The native true-residual suite passes51,866checks, including ill-scaled nonsymmetric local factors;12selected regression checks pass. The actual Human Schur path now accepts a root with mandatory full-operator certification and terminal checks, but takes21.907266s versus16.719385583s for matched uncondensed graph mode. It fails the existing1%comparison only at maximum anchor reaction (1.5378%);16other metrics pass. Tightening the existing mechanical tolerance exponent from3to4 does not resolve it: full21.777252208s versus Schur26.312927833s, anchor error1.4555%. Neither variant is an equivalent-algorithm qualification or a speedup.

Refined basis labels are now joined back to original anatomical donor IDs using exact source-tetrahedron order, rather than treating synthetic basis-region IDs as source identities. The new ancestry fixture and native app compile pass. The two earlier endpoint recipes fail before GPU work under the old ancestry check; later full-reference variants fail nonlinear convergence, and their receipts remain. An exact retained minimal accepted full-coordinate configuration with endpoint capture supplies a valid125us reference. Its source identities, mass, connectivity and rest pose exactly match the combined hybrid at14,202nodes/54,133tets. The combined hybrid still fails physical-force and regional strain gates (global volume-weighted strain relative-L1 error0.05291584, maximum absolute Frobenius error1.34845809 in left optic nerve tet92). Whole-body averages do not qualify the local errors. No reduced-model fidelity or speedup is claimed.

A passive-FEM rate-exponent0 trial using the minimal accepted solver completes four roots in7.52687275body seconds for0.5ms simulated (0.00006642865x), but differs from the retained rate2reference by7.44%in maxJ,10.32%in max displacement,78.05%in max anchor reaction and36.55%in contact normal force under the existing symmetric1%comparison. The separate more complex preconditioned rate0configuration fails nonlinear convergence before its first root. Neither permits a timestep/subdivision change.

An experimental preconditioner-prefix ICB implementation compiles with its flag off, but review identifies a diagonal-producer/cavity-Schur ordering defect and a missing field-boundary resource declaration. It is not published in these source blobs or credited as a gain. The prospective global-coarse Schur RAP fix is also not applied: that branch is excluded by current Schur admission and cannot explain the measured run. Actual selector6operator/preconditioner consistency and complete dispatch suppression remain ongoing work.

0.1x, supported periodic rest and sustained integrated physiology/presentation qualification remain unmet.


## Build50 minimal preconditioner graph and measured scope

The corrected opt-in preconditioner-prefix ICB admits the actual minimal selector0 Human configuration. Its FEM diagonal producer remains before the cavity-Schur consumer, and field boundaries/regional inverse resources are explicitly declared. Eight prefix and five tail commands are admitted; true-residual certification and acceptance remain outside the suppressible graph. Static contract checks pass15/15;12 selected regressions pass.

Three matched four-root release-transient pairs pass all17 comparison metrics and FEM/fascia terminal audits. Controls8.203005042/8.141068333/8.183159500s versus prefix7.874720375/7.919352500/7.878488125s give median1.038671300x throughput (3.87%). Candidate mechanics RTF is0.00006314–0.00006349x. This is not sustained coupled performance or0.1x qualification. Invocation receipts confirm all three required ICB/empty-field flags; the initial externally requested flags were cleared by the old runner and that trial is correctly retained as the off control.

The graph-only Schur ablation omits the existing12-mode partitioned preconditioner while preserving the nodal inverse, graph action, vascular/cavity path and mandatory full certificate. It passes17 metrics against build47Schur but is slower:24.270083584s versus21.907266s for125us (0.902645x). It remains an opt-in diagnostic, not a performance default. The next cohort repairs due-row consistency in local Schur matrices; this source publication does not include that later change.


## Build51 due-aware Schur local operators

Local graph matrices used by Schur condensation now use the same adaptive/microtick due-row predicate as the production residual and full operator. Non-due interiors have identity padding with zero RHS; source-sharing/interface nodes remain global. Ordinary graph preconditioning is unchanged. The native true-residual fixture passes, including a two-object different-period mass/zero-row case, nonsymmetric condensed RHS/recovery and original full residual. An initial unused fixture allocation fails the warnings-as-errors build; the exact failed log is retained and the allocation is removed before the passing build.

The Human Schur run accepts125us with unchanged full certification and terminal checks in16.344913541body seconds. It matches all17metrics of the previous21.907266-second Schur trial; the single unmatched-timing1.3403x ratio is provisional. Against the uncondensed graph reference it still fails the anchor-force1%gate by1.5378%;16other metrics pass. Thus this correctness repair does not qualify Schur equivalence or0.1x. Minimal selector0/prefix configuration remains preserved: four roots in7.922954583s with both terminal audits passed.


## Build52 conservative surface-order reuse

The opt-in surface Morton-order cache validates exact active primitive membership independently per environment. A hit reuses only the complete slot permutation; every primitive bound and every BVH leaf/internal bound is rebuilt from current geometry. Active-set changes force sorting. A cached snapshot is not rescanned on a hit. Production keysA reuse is admitted only without its Armijo alias; MPM/grid domains remain excluded. This cache owns no physical state or accepted histories.

Native contact checks pass7,772assertions, including large centroid reorder with identical active sets, two-environment invalidation, empty/full sets, duplicate bounds, exact-capacity2,556pairs and candidate-set/refitted-bound comparison against fresh sorting and a CPU AABB oracle. CPU incidence checks pass98,374assertions.

With the validated prefix ICB enabled in both arms, three four-root pairs pass all17 numerical comparisons plus FEM/fascia terminal checks. Controls8.477884750/7.913992000/7.914514625s versus cache7.646696792/7.630192959/7.673459792s give median1.037194215x (3.72%). The first control is noticeably slower than later controls; the median and all raw timings are retained rather than promoting its10.87%ratio. Candidate short mechanics RTF is0.00006516–0.00006553x, at least1,526x below0.1x. There is no sustained supported or native-presentation performance claim.

Fourteen selected compiler, reduction, contact and transaction regressions pass in the completed build52 source. The matched full-operator certificate control takes10.009128666body seconds for four roots (0.0000499544x); mandatory certification changes convergence and must be held fixed in subsequent warm-start comparisons. It uses4356Krylov columns across165active Newton begins. The existing EW tightening2 candidate uses3694columns across168begins and takes9.312822541s, but fails the1% gate for maxJ(1.9536%) and anchor force(4.2481%); fifteen other metrics and terminal audits pass. This is a rejected approximation comparison, not a qualified7.48%gain.

Verified APFS cloning of another16completed duplicate outputs preserves all paths and hashes and recovers2.22GiB actual available space. The exact incremental manifest,14-test log, completed build52 raw metadata and full-certificate control are retained in build52-completion-metadata.tar.gz with independently reopened per-member SHA-256 verification.


## Builds53–55: reduction fidelity and accepted-correction prediction

The refined compiler basis preserves anatomical source-face ancestry and adds fine coordinates around the left optic-nerve witness. It emits 1,872 reduced coordinates for 24,129 selected physical coordinates, plus 17,667 fine-complement coordinates. This is a 12.9:1 reduction in the selected region, stronger than the intended fourfold target. The complete Human reduced solve passes its terminal status but fails engineering fidelity: global strain relative L1 error is 5.2535%, skin/fat relative L1 error is 99.9985%, and the worst deformation-gradient difference is 4.2845 at left optic local tetrahedron5855. Both source-matched captures are retained. This configuration remains experimental and is not an accepted body model.

Sparse supported-mode Gram accumulation removes repeated dense startup work in basis assembly and whitening. The independent dense oracle checks every matrix entry, including cancellation and cross-region terms, with a floating-point roundoff bound. The source-label checker and fourteen selected regressions pass.

The opt-in accepted-correction predictor reuses at most two accepted Newton corrections within one physical microstep. It evaluates their proposed initial guess with the current full operator and solves the remaining residual. Original nonlinear norms, forcing scales and final true-residual certification remain mandatory; history is reset on each microstep/rejection. The native fixture passes52,206 checks, including nearly dependent history, multiple reduction tiles, original-versus-warm norm separation and repeated certificate checks. Fourteen selected regressions pass.

A matched build55 full-certificate pair takes9.834527417 seconds without the predictor and8.949999584 seconds with it for0.5ms simulated. Both terminal audits pass; sixteen of seventeen endpoint metrics pass, but anchor reaction differs3.337%. The provisional1.09883 timing ratio is therefore not a qualified gain. A tighter nonlinear comparison follows rather than promoting this result. No prediction gain is included in the accepted performance total.

Correction to prior wording: the fast minimal benchmark uses preconditioner selector0; selector6 is the separate block/coarse configuration. Invocation receipts are authoritative.

The tighter nonlinear exponent4 pair also fails the anchor-reaction gate (2.2812%):11.985131084 seconds control versus11.054152209 predictor. Sixteen other metrics and terminal audits pass. Exponent6 attempts fail admission because the supported range is0..4; their failed receipts are retained. No tolerance or admission was relaxed. The predictor remains off by default.

A second verified incremental APFS clone pass preserves22 duplicate files and all hashes, recovering3.056GiB actual available space. The new manifest and cleanup recipe are retained; no source asset or unique result was removed.

## Builds 56–63: contact interiors, full physical residuals, and reduced-coordinate work

These increments do not establish 0.1x, supported rest, or integrated viewer throughput.
Builds and authoritative checks in this section run on SSH Numi (Apple M6, 12 GPU cores).
The earlier best qualified short full-coordinate mechanics result remains 0.00006553x.

- Build 56 removes potential-contact-only anchors from Schur interiors, then promotes every patch touched by the actual compact active-contact corners back to the interface. It preserves all other attachment, field, vascular and cavity anchors. Build 57 telemetry reports 5,294 interior nodes in 576 eligible patches and 64 contact-promoted nodes at the last linearization. Malformed input now has an overflow-checked atomic-min witness and one deterministic failure publisher; nonzero contact slots and corners are covered by the native fixture.
- Build 57 timing is excluded because a reference process briefly overlapped the candidate. The overlapping reference was stopped and a sequential reference retained. Against that full-graph reference the active-contact Schur candidate fails min-J and anchor-reaction comparison. The geometry fidelity report retains the fascia-2 tet-59 witness. This feature remains experimental.
- Builds 60–61 replace the quadratic region search for each hybrid q mode with direct coefficient scatter under the compiler-validated bijection. It is selectable with NM_FEM_HYBRID_DIRECT_Q_SCATTER=1; the generic implementation remains the default and oracle. Three same-build pairs pass all 17 endpoint metrics at the existing 1% gate. Generic body times: 11.372457667, 11.278446625, 11.316874416 s. Direct: 10.564777500, 10.706258000, 10.601288500 s. Median paired throughput gain is 1.067499900x. Each run advances only 125 microseconds; this is an equivalent-algorithm gain inside an inaccurate experimental reduced model, not full-Human qualification.
- Hybrid diagnostics now preserve the original full RHS and measure lifted full physical b-Ax before restriction, separately from the q residual. A production-kernel fixture detects a residual orthogonal to the reduced span. The build-60 trace has 96 final cycles: the last full residual is 3.62202372e-5 against RHS 3.62490573e-5. This demonstrates real stagnation; 6,144 active columns are not merely encoded tails.
- Build 62 corrects a Schur stopping-scale inconsistency: Arnoldi keeps the condensed RHS, while forcing/full-space certification use the original physical RHS norm. Its native production begin/finish fixture passes. The Human result still fails equivalence (min-J 4.84%, anchor reaction 1.41%) and costs 27.120566208 s per 125 microseconds. This correction is not a claimed speedup.
- Build 63 retains registered skin layer 0 in the fine complement and propagates that source-node mask into the final-world rebuild. The fine source subset, compressible free span and whole solver dimensions are kept distinct. No fourfold whole-model claim is made. Source ancestry, existing optic boundary/witness policy and conservative mappings are retained.
- Internal constraint-response row/factor algebra is shared by Stand and coupled Human, with a separate native Metal fixture. This refactor does not admit source contact/equality constraints into the coupled Newton system. The existing admission restrictions and timestep guard remain.

Fifteen focused CPU/native regressions pass after the combined build-61 changes; the Schur normalization and layer policy have additional targeted checks. Full binary/source snapshots and raw short-run metadata are retained. Build54/build60 comparisons are not equivalent-configuration performance pairs: the former explicitly used mass whitening and the latter did not.

Build-63 actual source-subset accounting is 4,714 fine nodes plus 1,872 q modes: 31,275 -> 16,014 source coordinates (1.95297864x). Whole-body coordinates are 41,796 -> 26,535. Both whitened and unwhitened layer-0-fine trials reject with status 10 before accepting any step. The retained fine rows survive lift/restrict/packing; no lost-row defect was found. This model is not qualified. The unchanged full-coordinate build-63 control passes both terminal audits, advancing 0.5 ms in 7.904341375 s (0.00006325638x mechanics).
The additional Schur domain-latch correction uses the original physical residual for domain sums as well as forcing/reference norms; its production native fixture passes after a retained fixture-scope compilation repair. A fresh integrated run remains required.
Verified APFS cloning of identical completed-run outputs restored 14 GiB available; the sibling-clone receipt preserves every destination/source hash. No unique source or evidence content was deleted.

## Build 64: physical Schur rejection and bilateral primitive

The original-physical domain correction now rejects the Schur Human trial before any accepted step: underlying Matter status5, object4, tet198 (myofascia2 local54), trial J0.72465. The wrapper horizon code is9. This is a retained failure, not a faster qualified solve. No threshold was relaxed.

The native bilateral KKT fixture invokes the production coupled kernel. It checks A0*deltaV+J^T*lambda=forceRHS, J*deltaV=constraintRHS, and unchanged outputs after a singular factor. It passes, alongside the shared constraint-response and FGMRES checks. A dedicated include keeps coupled ABI dependencies out of the Stand shared algebra. This primitive is not yet connected to the running Human; source-contact admission stays closed.

The hybrid direct-scatter setting is parsed and validated once at initialization instead of querying the process environment per encoded column. Its default remains the generic path.

The same full-coordinate control passes terminal FEM/fascia audits and all17 endpoint metrics against build63:0.5ms simulated in7.932955125s, RTF0.00006302821784. Phase17 profiling has three invalid radix-prefix timestamp records, retained explicitly. Observed root0 stage intervals total398.405ms assembly,379.876ms linear and77.878ms globalization; nested kernel samples are not scaled or added into whole-run shares. This establishes neither sustained rest nor0.1x.

Verified duplicate-output APFS cloning restored14GiB available, without deleting unique content. Exact source changes, artifacts, failing build attempts and run receipts are in build64-native-metadata.tar.gz and its independently reopened manifest.

### Build65: repair Schur restart RHS; retained failures and larger-step probe

The exact interior Schur RHS is re-condensed after each true-residual restart.
The strengthened native fixture constructs the implicit Schur columns on GPU,
solves the interface system, recovers interiors, and compares against the full
dense solution. The uncondensed-RHS negative control fails as expected.
Native fixture passed (0.51 s). No acceptance or residual tolerance was loosened.

The full Human now accepts one 125 us root and passes both terminal tissue
audits, where build64 Schur rejected before any accepted step. It is slower:
23.311168 s versus same-build full-coordinate 18.978371 s. The existing numerical
comparison fails the 1% gate: maximum anchor reaction differs by 2.22886%.
Endpoint strain/force checks also fail (fascia2 force relative L1 27.856%).
This is an opt-in solver correctness repair, not an admitted performance gain.

Accepted-Z0 recycling build64 repeats pass terminal audits but fail max-J parity
(1.6488%). The run named recycle-control-4 actually has recycling enabled;
the authoritative invocation records this. The genuine off control is
historyoff-full-4. Their approximately 4.3% timing improvement is unqualified.

The build65 full-coordinate 0.5 ms step passes terminal audits in 3.462661209 s
(body-only 0.0001443976x). It remains a single release-transient step, not
supported rest or sustained throughput. Final measured nonvascular residual
7.79657e-5 exceeds the tighter domain-latch target 1.25e-5; the existing terminal
status alone does not qualify this timestep or the 0.1x requirement.
Raw receipts, logs, failed comparisons, frozen source and artifact identities
are retained in build65-and-late64-native-metadata.tar.gz with an independently
reopened per-member hash receipt. No settled/coupled/viewer claim is made.

## Builds 66–67: operator audit and honest initialization admission

The opt-in `NM_FEM_SCHUR_AII_OPERATOR_AUDIT=1` measures the unfactored current fascia-2 interior block against the authoritative coupled operator. It requires selector-6 Schur, active-contact eligibility, and one environment. Build 67's two probes returned relative L2 errors 1.35816705e-7 and 1.99066719e-7, both operator status 0 and invalid 0. All 17 endpoint comparisons with build 66 passed. This is a bounded patch/two-direction check, not whole-Schur equivalence or a speedup.

`NUMI_HUMAN_SUPPORTED_PRELUDE_STEPS` adds initialization to the same continuous GPU-owned transaction and requires eight consecutive full-gravity accepted states satisfying support, acceleration, speed, contact, conservation and owner-clock checks before measurement. Default zero preserves the existing path. It neither resets controller history nor injects support forces.

The 40-step prelude with the existing 33-step gravity ramp and declared 101.2 micrometre initial clearance rejected physical step 37 after 36 accepted steps (4.5 ms). At step 36 support reaction was zero, root speed 0.377433 m/s and maximum generalized acceleration 347.666 m/s². Matter rejected object 0 with diagnostics [0.0008239331, 0.00008816063, 0, 0]. This is a failed initialization, not a supported checkpoint; no sustained performance or timestep qualification follows.

A 0.5 ms one-root sensitivity passed but continuation rejected its third root after 1 ms. The 0.125 ms reference continuation rejected after 14 roots/1.75 ms on vascular Armijo. These failures are retained in build65-timestep-continuation-failures.tar.gz and its independently reopened per-member receipt. The larger step is not qualified.

Build 66–67 source changes, exact artifact manifests, raw native receipts, failed preparations, build logs and passing operator audit are retained in build66-67-native-metadata.tar.gz (SHA256 9e125b311ed0a70e886ba97779923e08666f699edda53693035ffcac4ed22a53). These changes improve diagnosis and initialization admission, not measured throughput. The minimum 0.1× integrated requirement remains unmet.

## Builds68–72: physical residual-guided refinement and assembled operator

The opt-in hybrid witness records full physical residuals before restriction, with immutable source identity and exact basis-layout hashes. The existing compiler consumes a validated trace to retain additional fine nodes, preserving attachments, contact rows and source ancestry. Candidates are ranked explicitly: 605 eligible records, 128 selected, 477 excluded by the declared global budget. Invalid/nonfinite traces, duplicate identities and insufficient coordinate budgets fail closed. Runtime admission now matches the compiler's existing minimum twofold coarse-space setting; default fourfold construction is unchanged.

The actual build72 refinement adds128 fine nodes but rejects before an accepted physical step. Its initial/final full residuals are3.29253991e-4 and3.26104229e-4, versus2.94759084e-4 and2.85422662e-4 in unrefined69. Projected residual also stagnates, so omitted fine complement alone cannot explain failure. Next certify the complete hybrid fine-plus-coarse operator/preconditioner rather than adding nodes blindly.

NM_FEM_ASSEMBLED_XYZ_OPERATOR=1 builds deterministic3x3 block CSR from the current exact constitutive cache, preserving mixed/invalid-cache fallback and an independent direct full-operator certificate. A stale previous-linear-solve completion latch initially zeroed new-Newton stiffness; build71 removes that guard and honors current Newton completion in both assembly stages. No acceptance tolerance is relaxed. The four-root Human passes terminal tissue audits and all17 numerical comparisons:9.8954245s off versus14.884046125s on. This is a regression. Root0 whole-stage profile records1740.694ms linear and327.763ms assembly, without scaling sampled kernels.

Build72 uses translation-stable off-diagonal actionKij*(xj-xi), enforcing the analytical FEM translation null mode. Its727 native checks pass: current bytecode tangents, two environments/timesteps, random directions, fixed rows, overlapping elements, stale cache lifecycle, completed-tail sentinels, translation/cancellation, force closure, nonfinite witnesses, invalid-cache fallback and empty domains. The first closure assertion failure was a retained fixture subtraction error; the repaired test starts from zero work without weakening its tolerance.

Build72 accepts four Human roots and passes terminal audits in13.124954875s, but fails max-J equivalence by1.47337%; other16 endpoints pass. It remains experimental. Parallel row action and compact deterministic edge-contribution assembly are the concrete next implementations.

Both compiler reduction regressions pass. Failed build attempts, native fixtures, raw trials, exact source/artifact hashes and trace-selection receipts are retained in build68-72-native-metadata.tar.gz, SHA256 1a7270c3a631f6debeaa66315008f8b2803c2a3efd1b91ad796a8ab7ad28b001. Every member was independently reopened and verified. Verified APFS cloning preserved completed-output bytes while recovering space.

No short release-transient run here qualifies supported rest, sustained coupled throughput, native presentation or the minimum0.1x requirement.

## Build 73: compact edge assembly and SIMD row action

The optional assembled FEM path now uses a deterministic compact node-edge-to-tetrahedron contribution map instead of rescanning node incidence for every edge. Construction validates topology, ownership, capacity and immutable-cache eligibility. Mutable FEM topology remains excluded. NM_FEM_ASSEMBLED_XYZ_SIMD_ROW=1 applies each CSR row with one SIMD group; direct and cached ICB dispatch dimensions both cover all node lanes. Both settings remain off by default.

The focused actual-Metal checker passes 995 checks, including bitwise edge assembly agreement with the incidence reference, serial/SIMD action, stale-cache lifecycle, fixed rows, cancellation, translation, empty domains and malformed shared metadata. The first malformed-metadata test incorrectly expected an unaffected second environment even though the contribution map is shared; the corrected fixture expects both environments to reject. The complete existing true-residual native suite passes 53,637 checks. These fixtures explicitly do not establish full FEM physical qualification.

Same-build four-root Human runs accept all steps and pass terminal tissue audits. Candidate body time is 10.733242333 seconds versus 9.892284458 seconds for the original operator (0.0000465842x versus 0.0000505444x). The candidate improves the previous experimental assembled path's 13.124954875 seconds, but remains slower than the original. Its maximum-J endpoint differs by 2.08418%, failing the existing 1% comparison; the other 16 endpoints pass. Further repetitions and longer qualification were stopped at that failed gate. This is tested opt-in infrastructure, not an admitted performance improvement.

Exact source and binary identities, both raw runs, failed build/test attempts, checker repair source, private patches and passing native logs are retained in build73-native-metadata.tar.gz, SHA256 0436b5e6c5211756168762cf70c38d144b4476dcf43c555257bacd2e27e5184a. Every member was independently reopened and hash-verified. The frozen dirty snapshot also records then-unpublished external-load work; that is not part of this sparse-operator publication.

Supported rest, sustained coupled performance and the minimum 0.1x remain unachieved. The next high-value work is full fine/coarse operator certification and correcting the real support/recruitment operating point.

## Builds 74–75: paired physical FEM load readout

The existing static recruitment owner accepts an explicitly identified generalized external load at a frozen source pose. Validation rejects stale model identity, dimensions, nonfinite values and pose mismatch; no-load callers retain the existing interface. This is an initialization input API, not a completed supported-rest initializer.

The optional NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE=1 path snapshots the actual FEM/body-contact generalized load (the separate bone-anchor reaction channel is not yet included) and exact source q before dynamics mutates source pose/Jacobian buffers. Final publication uses the existing exact accepted Stand status and completed-step gate; rejection zeroes both outputs. It adds no force. The accessor and native diagnostic return the paired pre-step q explicitly. Default-off avoids its buffers and pipelines.

The native capture fixture mutates source wrench/pose/Jacobian/q after preparation and verifies accepted publication keeps the snapshot, then checks stale/rejected publication. It and existing static support tests pass (0.26 and 0.52 seconds). Build74's four-root actual Human capture-on run passes both tissue audits and all17 endpoint comparisons with build73 control. Body time10.0965245 seconds (0.0000495220x), not a performance gain. Build75's one-root actual owner readout publishes129 source coordinates and128 generalized forces; initial contact-load norm is0.0121108464 in generalized units, with root components[-0.00242410,0.01098807,0.00283480]N.

The -101.2 micrometre initial-root placement experiment passes four steps but is not settled: root speed0.91081m/s, maximum generalized acceleration509.58m/s² and82.07micrometre penetration after0.5ms. This translates freshly constructed stress-free geometry against the unchanged bed, not a valid precompressed checkpoint. The generic support_force_n measures articulated support rows; it does not measure total FEM-bed reaction.

Raising recruitment's activation cap from0.01 to1 exposes a separate admission failure before integration: initial normalized residual approximately7e-6 deteriorates to0.005413 during optional stop-load unloading, despite a much smaller0.222472m/s² acceleration residual. The final<=initial gate correctly rejects it. The frozen-pose secondary objective policy needs repair; no tolerance is relaxed and no dynamic result is claimed for that trial.

Exact source/artifact manifests, raw trials and failed build attempts are retained in build74-75-native-metadata.tar.gz, SHA256 c8bdf48af37d5c5006c192da2c3842172ac5a82c9bdac47cdcea4a736dfca29a; members independently reopened and verified. These are readout and initialization diagnostics. Supported rest and0.1x remain unmet.

## Build76: preserve physical balance during frozen-pose recruitment

The app now derives optional joint-stop unloading's residual-increase policy after all pose-sweep overrides. A frozen-pose solve retains the existing1e-12 nonincrease allowance even without explicit static supports; an actual posture search retains its previous balance-bounded policy. The physical laws, complementarity certificate and final admission gate are unchanged. The static support regression explicitly covers zero pose sweeps and passes.

With the full0–1 activation recruitment range, the same-pose Human now gets through initial admission: normalized residual7.15244989e-6, maximum acceleration residual7.76028191e-5, versus123.27933 and531.33014 for the old0.01 cap. This is internal fixed-root recruitment balance; root gravity was not included and the log still classifies it as a release transient. Four physical roots accept and both terminal tissue audits pass. Body time11.491004834seconds,0.0000435123x. Root speed0.43116m/s and generalized acceleration250.72m/s² after0.5ms demonstrate that it is not supported rest. The separate contact readout grows to root vertical force-2.8378N by source step3; bone-anchor reaction loads are not yet captured by it.

The native Human target builds. A wider generic probe build fails on pre-existing unguarded resting transaction-probe types; recompiling exact HEAD source reproduces that failure. It is retained separately and is not described as a passing full build.

Frozen source, raw run, passing static test, failed generic build/control and policy patch are retained in build76-native-metadata.tar.gz, SHA256 c1b735817c15f56dc285d26bf4daf49a793facf1cbc2044366ce08e7d20cb6cb, with independently verified members. No supported or integrated-throughput qualification follows.

## Builds77–78: total tissue-to-skeleton readout and coupled smoke

The optional accepted-load capture now adds the fresh post-Matter bone-anchor reaction projection to the body-contact wrench snapshot before articulated force application. The physical force path is unchanged. No stale pre-Matter anchor buffer is used. Exact source q stays paired with the same pre-Stand transaction, and rejected/stale completion zeroes both outputs. The fixture covers body-only, anchor-only (including ignored NaN body data when there are no body contacts), combined contributions and rejected transactions. Native capture and static-support tests pass.

Four-root actual Human capture passes both tissue audits and all17 endpoint comparisons against build76, with11.4108565seconds body time (0.0000438179x). At source step0, total root FEM force is[-0.0197143,0.881935,-0.180227]N; by step3 it is[1.29604,-1.62596,-4.80661]N. These small forces confirm no initial physical support balancing the rigid body's weight. This is a measurement, not a supported operating point.

A non-mechanics-only native smoke accepts one125us step, passes both tissue audits and reports matched physiology/body clocks without root assistance. It runs the existing coupled respiratory Brain program and presentation observer. Body transaction time3.536102792seconds, observer42.14ms including30.83ms presentation. This single-step smoke is neither sustained integrated performance nor visual verification; no images were inspected.

The unmodified hybrid operator retested with cap1 still rejects before any accepted step in33.0616seconds. Its first and final physical residuals exactly match build69 (2.94759084e-4 and2.85422662e-4), isolating its stagnation from the recruitment-cap problem.

A partial checker77 fixture passes1207 checks and exposes nonzero fine/coarse cross blocks through the GPU FEM action/lift; its Arnoldi orthogonality error is1.32406e-8. Its restriction and dense solve are host oracle operations, so it does NOT establish production hybrid restriction, preconditioning, restart convergence or physical equivalence. A full production cycle fixture is still required.

Exact sources, artifact hashes, raw native runs, compact CSVs and failed checker-build attempts are retained in build77-78-and-hybrid76-native-metadata.tar.gz, SHA256 6e56ba2628ce10e578edc77a6b33e8ed32a2b94f6e4374495a886ff286289b36, independently reopened and member-verified. This publication includes only the total-load readout; the partial hybrid checker remains pending its complete solver test.

## External FEM load: floating-root certificate

An explicitly supplied, frozen-pose FEM generalized load now includes the floating-root force and acceleration rows even when no synthetic support contacts are supplied. Internal fixed-root balance is no longer sufficient for this external-load API. This does not create support or supply a force.

The static owner regression passes: the exact gravity-balancing load satisfies the root certificate with zero synthetic support rows; removing that load fails balance. A first fixture build attempted to mutate a const span and was repaired with owned storage. Its first run also exposed a 1.05e-6 N mismatch between double-literal gravity and the actual Float32 source gravity; the independent oracle now uses the exact source value, with unchanged tolerances. Failed attempts and passing source/logs are retained in external-root-certificate-native-metadata.tar.gz with independently verified hashes. This is a certificate repair, not supported-rest or performance qualification.

## Builds79–81: hybrid block coupling, complete cycle tests, safe diagnostic rejection

Added opt-in NM_FEM_HUMAN_HYBRID_BLOCK_GS=1, using the existing current FEM element-mode actions and selector-6 factors for multiplicative fine/q corrections. For nonsymmetric tangents its cross-block transpose is explicitly an approximate preconditioner; the full physical operator and mandatory certification remain unchanged. Admission requires the existing hybrid coordinates and FEM tangent path. Defaults are unchanged.

Native production-kernel fixtures pass:77 hybrid GS checks,2059 assembled/hybrid checks including a forced two-cycle GMRES(1) restart reaching certified relative residual8.15801e-7 at authored1e-6 forcing, and53711 checks in the complete native true-residual contract. The tighter1e-8 three-column fixture reaches5.75705e-8 and is retained as an uncertified Float32 roundoff control. Host element gather remains an explicit fixture limitation.

In the actual Human cap1 release trial, GS reduces the final domain residual from2.90099793e-4 to1.15786905e-4, but still fails the unchanged1.25e-5 physical threshold. It accepts zero steps, spending41.66698725seconds in the process with24 active Newton iterations and1536 active Arnoldi columns. Mass whitening produces1.35016890e-4 in42.42186775seconds and also accepts zero steps. Existing full-coordinate global coarse accepts one root but fails terminal optic-nerve geometry(J=.278237); no physical or performance qualification follows.

A misconfigured residual diagnostic exposed an encoder released without endEncoding on its admission failure. The guard now closes the encoder before rejecting. A negative trial without the required true-residual root selector exits1 cleanly after18.388904seconds and zero accepted steps, reporting the intended admission error. Earlier aborts and an accidentally enabled tangent-witness diagnostic remain retained.

Exact frozen build79–81 source manifests, raw trials, failed builds, passing test logs and source patches are retained in build79-81-hybrid-gs-native-metadata.tar.gz, SHA25623801e606fe03fdc08e1dcce58f22bf5ef2dca5fec092ff4d32817ddce8ba9a2, independently reopened and member-verified. These are tested solver increments and failed Human experiments, not supported-rest,0.1x or sustained integrated qualification.

## Builds82–94: Schur consistency, hybrid convergence and failed fidelity gates

The Schur graph/interior map now uses the original graph-patch identity rather than assuming a compact block index is that identity. Native tests explicitly use nonidentity IDs. Raw interface action agrees with the full operator at roughly 1e-7 relative through all active Newton solves in the retained Human trial. The separate per-Newton audit now classifies completed or failed tails as skipped, rather than falsely comparing stale work. Native raw-action/audit checks pass167; the complete build94 native solver contract passes53,876 checks. The native checker now rejects unknown CLI options before executing unrelated tests.

The corrected raw-action Human trial still rejects before any accepted step, at vascular row32, in32.757650667 process seconds. At microtick1/Newton20, the aggregate vascular merit directional derivative is +5.25846389e-9; initial merit2.01722639e-10 becomes2.01957451e-10 at alpha6.10351562e-5. Its witness linearization error is2.18051355e-8. Positive aggregate slope establishes a non-descent vascular direction; this is not established as a floating-point floor and no Armijo criterion was weakened. Enabling the existing vascular predictor instead fails tissue geometry at object4/tet203. Both failed trials remain retained.

A same-frozen-build91 hybrid GS comparison isolates the graph Schwarz option: graphON accepts one125us root with both tissue audits passing in52.06769625 process seconds; graphOFF rejects before accepting a step in40.799566417seconds. The extra global six-mode correction is unnecessary for this particular accepted root. This is a convergence observation, not a speedup. Prior runs using misspelled environment controls are retained but excluded from causal comparisons. The checked runner now rejects unknown solver environment controls before launching.

Build94 captures both full coordinates and graph-Schwarz hybrid GS at matched125us endpoints. Both accept one root and pass terminal tissue audits: full32.810480417 versus reduced52.650486875 process seconds. Their source identities, masses, connectivity and rest poses agree. However the existing engineering comparator FAILS regional force and deformation gates: left/right ACL integrated force relative-L1 errors0.54849/1.25421; global volume-weighted strain relative-L1 error0.002709, but skin/fat and both optic regions fail regional gates. Small reference strains and force magnitudes require converged reference comparisons; these failures do not establish the approximation as qualified. Physical force-owner coverage in this capture is also incomplete. No reduced-model or performance acceptance is claimed.

The completion-only hybrid diagnostic reconstructs the projected residual from explicit q/fine/pressure/MPM/rigid/support/vascular domains, with finite/closure checks. In the retained root, the fine complement dominates (9.3530e-6 versus q1.0350e-7), and inactive padding is zero. The diagnostic is observational and does not affect solver acceptance.

The row-equilibrated raw-Schur experiment is rejected by the existing explicit unsupported-combination gate; no physics executes. That restriction remains intact. The new conditional selector6 sparse-plus-graph prefix is under native fixture validation and is not yet a qualified Human option.

Supported rest, timestep qualification, sustained coupled throughput, 0.1x, and the long baseline/intervention remain unmet. All mechanics trials above are release transients; none establishes supported equilibrium or integrated presentation throughput. Exact frozen manifests, source patches, raw runs, failed trials and native logs through build94 are retained in build82-94-schur-coarse-native-metadata.tar.gz (SHA256134c782d00942a3f21abf58ca378a636b637ff9f25aa260846f5d9a0747875f2), independently reopened and member-verified.

## Resource recovery

Verified idle macOS GPU-archiver staging caches had accumulated43.5GiB. With no readers,65 old cache directories were removed after per-directory age/type/lsof checks:21,403,743,872 logical bytes; live free space rose from about0.4GiB to20.5GiB. Sources, current evidence and the retained pipeline archive were preserved. Twenty-five older replay files were retired only after GitHub release upload, independent download, archive/member SHA verification and a restoration check. Manifests and retirement receipts remain retained. The12GiB qualification storage requirement is now met.

## Builds95–98: conditional selector6 prefix validation

The new six-command sparse/graph subgraph fixture passes281 native checks: two column buffers, iteration-constant refresh, nonzero active work, bitwise direct/indirect agreement and an empty converged range preserving scratch. The complete native solver contract passes54,154 checks. This fixture covers the added subgraph, not all15 commands in the production Human prefix.

The first fixture and Human opt-in run crashed while assigning ordinary PSOs to indirect commands. The fixture now creates indirect-capable PSOs explicitly; production pipeline admission now includes nm_fgmres_apply_coupled_tangent_block. Runtime command access verifies graph-specific capacity before indexing, and a capability guard rejects an unsupported PSO with its name before calling Metal. A source contract checks every statically recorded prefix/tail kernel against the indirect classifiers. Its16 tests pass; two stale capacity-string assertions were updated to the new named capacities after retained failures.

The repaired build96 Human run records15 prefix and5 tail commands and reaches exactly the same physical rejection (vascular row29, identical four diagnostic values) as the direct-prefix control. Neither accepts a step. Process times25.758707834 versus26.846314583 seconds include startup, differ in source revision and are not a measured throughput gain. Preferred vascular principal/cavity prefix composition remains excluded pending explicit dependency/parity validation.

Build95–96 exact frozen sources, native tests, invocations, failure traces and crash report are retained in build95-96-schur-coarse-native-metadata.tar.gz, SHA256c6541cce44cc008ba06f52f4aa0d760cb64ab559b6f5aee9c9be1b24d8efe8f5, independently reopened and member-verified. No physics or acceptance threshold was weakened. The frozen-q static-load evaluator is a separate unqualified work-in-progress increment and is excluded from this source publication.


## Builds98–102: coherent supported construction and vascular prefix

The strict registered-skin stance fit was exercised over lower-hull contact triangles. Rows0/1/2 and1/2/4 fail the unchanged static balance gate (maximum generalized acceleration residual8.806623 and1.622457m/s²). Rows1/4/5 pass the strict full-gravity articulated recruitment gate with three closed, non-collinear skin contacts. Fitting this root after the existing FEM construction initially inverted fascia object2/tet74 (candidateJ=-0.128254, acceptedJ=1); the failed trial is retained. Recompiling the reference tissue at the recorded fitted root placement clears this immediate frame mismatch. This is a declared fixed resting construction, not a settled full tissue/vascular equilibrium checkpoint. No joint-angle edit or root assistance is used.

Build100 then accepts one fully coupled125us step with matching physiology/body clocks, terminal tissue checks and body execution17.678392042s (0.000007070779x). Build102 independently accepts matched direct-prefix and conditional-prefix steps from identical input root placement. Their compact physiological/contact trace numerical fields are identical. Body execution20.972584416 versus20.257491625s is a single matched pair (ratio1.0353), insufficient to establish a sustained performance gain; process time includes startup. The conditional preferred Human graph now includes the vascular diagonal and principal-block consumers, while cavity-Schur diagonal and principal-factor setup remain direct. Predictor, unsupported hybrid/Schur combinations and acceptance restrictions remain enforced.

The new vascular-prefix fixture checks two vector columns and two timestep/cavity-diagonal configurations, unchanged nonvascular rows and an empty converged range; direct/ICB outputs match bitwise. The full native FGMRES kernel suite passes54,327 checks and16 source-contract tests pass. These are solver-kernel checks, not full Human physiological or performance qualification.

A bounded frozen-q static FEM load evaluator reuses the production internal, cavity/tissue-pressure and gravity force paths without advancing accepted state. It rejects active contact, cohesive, mutation and MPM configurations and therefore is not yet a complete Human support wrench. Its two-environment64-node fixture passes global force closure, external-force exactly-once accounting, fixed-boundary reactions, ordinary snapshot immutability including typed sentinels, content-seal rejection and declared pressure-phase rejection. Failed builds97/99/100/101 are retained: they exposed optional sentinel shape handling and two fixture comparison/message errors, which were repaired without weakening runtime checks. Sources that explicitly disable both deformable-contact modes no longer reserve implicit default contact capacity; explicit source reservation remains authoritative (tested reserve7).

The supported1ms trial fails certified edge–edge narrowphase (object19, candidate4393, primitives16050/16048, edges2/0), before any accepted step. This is CONTACT_FAILURE code6, not capacity overflow. Failure endpoint capture/replay is underway; no CCD budget, tolerance or rejection gate has been relaxed. The20 explicit skin/fat springs still block2ms; exact candidate-articulation and implicit attachment integration remain in progress. Supported periodic settling, qualified timestep growth, reduced-model accuracy,0.1x integrated throughput and10/60/330s qualification remain unmet.


## Builds103–123: exact candidate ownership, contact certificate and graph measurements

The ABI-7 borrowed transaction bridge now evaluates candidate source articulation on the GPU from a source checkpoint, with explicit operation admission, typed buffer bounds, FK/Jacobian provenance and rollback. Native candidate FK, candidate rollback, legacy accept/rollback and ABI probes pass. This bridge does not provide pure articulated mass action, candidate-q mass inversion or staged reaction publication; the20 implicit skin/fat rows remain disabled. Source commitc2da83ca is pushed to main.

The exact Human EE CCD failure is now replayable. The well-conditioned segment certificate no longer spends the same distance-roundoff margin twice; nearly parallel cases keep the stricter margin and fail-closed treatment. The bounded EE interval cap is96, while VT remains64. The captured clear case requires71 interval checks; the replay retains its cap64 uncertainty and passes true-impact, near-parallel, reversal, primitive-permutation and translated-clear controls. Source commite40ea89f is pushed to main. The1ms Human now proceeds beyond this specific CCD failure but rejects nonlinear convergence before accepting a root; it is not a qualified timestep.

Full per-call graph timing with65536 slots avoids the earlier truncated4096-slot profile. Build113 measures graph matrix construction6.567696757s, factorization0.527609546s and6144 encoded patch-solve calls6.392872369s inside17.18860479s total linear-stage GPU time. These6144 calls do not establish6144 active iterations: used-column telemetry separately records1693 columns over52 active Newton begins in the matched short cohort. Nested kernel times must not be added again to the whole linear stage.

The first parallel patch substitution (build114) lowers solve time to1.409659042s and body time21.029253375→15.922607208s, but fails the unchanged1% endpoint gate (maximum acceleration4.02%, anchor reaction3.06%, root velocity1.34%). Its upper-triangular accumulation reversed the legacy order. The corrected order-preserving build119 passes all17 endpoint metrics against113; graph solve time is5.26205s and body time20.051163958s, a single short pair and not a sustained speedup claim. The55558-check native solver contract includes an independent legacy compensated-substitution oracle.

Graph tet-block reuse is opt-in and remains unqualified: build119 matrix construction6.61893s offers no useful measured improvement versus6.64495s without reuse, and maximum generalized acceleration differs1.75%. The path currently excludes mixed FEM. Its new production-generator-to-graph test initially failed because its fixture bound a one-float dummy to the block input; the corrected real-buffer fixture passes2367 assembled-operator checks, including modes0/1/2/3, three geometries, two environments and nonuniform dt. Fixed-row coverage is being added. This fixture success does not override the failed Human comparison.

Node-level Schur contact promotion now preserves unrelated interior coordinates rather than excluding whole contact-active cores. The Human admission retains581 cores/5358 interior nodes, but build119 still rejects at right-ventricular vascular row16: base merit3.056397135e-9 becomes3.056402020e-9 at alpha4.34253195e-7. A cross-branch Armijo experiment retains actual residual evaluation, positivity, thresholds and an exact representable-breakpoint fallback; both directional-valve regression paths pass, but it does not fix this Human failure. The original failure is retained. The branch experiment is off by default.

A proposed graph-only preconditioner ICB extension was built as123 but withdrawn before measurement: the preferred partitioned graph profile already records those commands. It would add no coverage to that run. Native compiler/probe fixture errors and failed trials remain retained instead of being promoted to gains.

Metadata/source patches/raw traces through123 are retained in build103-123-solver-contact-native-metadata.tar.gz, SHA256d96ebc83580b926e6f84a0f21300f533234d7f4acaa20c5621e04f1a496e6c87 (7882307bytes); every member was independently reopened and hash-verified. Current accepted short body throughput remains approximately6.23e-6x, not0.1x. Supported full-tissue equilibrium, qualified reduction/timesteps, implicit20 attachments, sustained10/60/330s runs and native presentation qualification remain open. No visual verification was performed.

### Builds 124–133: certified controls, graph reuse and vascular globalization

Full coupled linear true-residual certification is now explicit in the preferred matched controls. Earlier profiles without that opt-in retained nonlinear physical checks, but were not full coupled linear certification runs. Build124 control: 16.88082 s body time for 125 us; patch16: 20.56792 s, all 17 endpoint metrics within 1%, no speed gain. These are short initialization steps, not sustained throughput or native presentation qualification.

The graph solver uses parallel right-looking updates while retaining each scalar's legacy pivot/update order. Native checks cover generated tet blocks, graph matrices, fixed rows, random directions and completed tails. New graph-only tangent-cache marker 2 cannot be consumed by the full operator. Cache reuse remains experimental: build131 control 17.40500 s versus reuse 23.20107 s, acceleration differs 6.25% and anchor force 6.12%; rejected by the 1% gate. Build133 remains approximately 17.31725 s control versus 23.15792 s reuse. The earlier ICB-disable hypothesis was contradicted: its assignment preceded a later initialization which overwrote it. This does not explain the regression. Do not promote block reuse to the accepted Human configuration.

The live Human census has 54,133 active FEM tetrahedra, all NONMIXED and stateless; 949 graph patches, 89,013 patch rows, maximum 32 nodes. The mixed-material extension therefore provides no direct Human speedup. It remains covered by a native fixture. The generated-block parity fixture passes 2,518 checks; the residual-contract fixture passes 55,558; the Python contract suite passes 17 tests.

Bounded alpha-only traces now cover the full certified Human without widening physical admission. The condensed solve still rejects with zero accepted physical time. At microtick 0, Newton 6, support alpha is 0.000889315677; vascular globalization exhausts its trials, ending at tested alpha 4.34236e-7. Its aggregate merit derivative is negative, but the measured trial merit increases. Contact/support stages are not responsible for the additional vascular reduction. Compensated vascular residual/merit arithmetic preserves the thresholds, passes native directional/legacy tests, and rejects a production-kernel near-ULP false acceptance, accepts a genuine decrease control, and fails closed on NaN. It does not resolve this Human condensation failure.

Source-M transaction service is separately published as 623cddee; vascular compensation as c062b02e. Source-M remains source-q only, tested on a six-DOF fixture; mixed-scale joint conditioning, transaction-local factor reuse and staged implicit body coupling remain ongoing. No larger timestep is qualified. No 0.1x claim is supported: build133 control achieved 0.00000721823x excluding startup, still a short initialization step. Sustained end-to-end throughput and 15 fps native presentation remain unqualified.

Archive build124-133-solver-contact-native-metadata.tar.gz SHA-256: 839d114bb60ff5337229e21f62ba5c8dfc8224cec40077adfaa7f0a6676f5e64. Every archived member was independently reopened and hash-verified. Failed builds, failed comparisons and Schur traces are retained. Idle rebuildable GPU archive cleanup restored approximately 34 GiB available; exact before/after df and deletion manifest are retained.

## Builds134–135: combined graph, vascular diagnostic, and source-mass work

Build134 replaces the graph cache path with a dedicated row builder that has no material interpreter bindings. The generic builder remains the fallback for incomplete caches. The effective ICB initialization now excludes only mixed cached graphs; the earlier guard was overwritten later and did not explain the Human mismatch. Human's 54,133 active FEM tetrahedra are all stateless and nonmixed, so the mixed-cache extension itself does not claim a Human gain.

The native true-residual suite passes 55,558 checks, the layout suite passes its bounded-allocation checks, and the production vascular GPU near-ULP rejection/control/nonfinite checks pass. Build135 extends generated-cache parity to dt=31.25 us with stiffness-dominated matrices and no unit-scale error floor: all 2,751 assembled-operator checks pass, relative matrix errors approximately 8e-8. This does not prove full-Human matrix parity.

The source physical-mass inverse now factors a diagonally equilibrated matrix without regularization and reuses a successful factor only inside its transaction. The ordinary six-DoF action/solve and rollback pass. After repairing missing sidecar probes in the adversarial fixture, the mixed mass/inertia seven-DoF action/solve passes with roundtrip error 4.77e-7; rollback and the 48-byte program ABI contract pass. ABI9 stage ownership is opt-in and is not yet an actual implicit attachment producer or full-Human qualification.

The memory/conditioning experiment retains the default 96 MiB graph arena and permits explicit checked 192/384 MiB arenas. Build134 control body time is 17.362774375 s for 125 us (RTF 0.00000719931). 64-node/192 MiB patches take 23.980940416 s and miss the 1% acceleration comparison (3.823%); 128-node/384 MiB patches reject physical acceptance after zero accepted roots. Neither is an optimization gain. No memory cap, residual, physical, or comparison gate is weakened.

Per-attempt vascular diagnostic records show an FD/JVP discrepancy before the smallest-alpha merit floor. At alpha 0.000889315677, row27 residual changes from 1.42822262e-6 to 1.83563236e-6 while its JVP is -1.42059582e-6. At half alpha, row28 likewise moves opposite its JVP. The condensed solve still rejects; this is evidence for further derivative/trial-state diagnosis, not permission to accept a nondecreasing step.

All execution remains SSH Numi. Native presentation and sustained 0.1x are unqualified. Full conditional solver subgraphs, actual implicit attachment coupling, supported tissue equilibrium, reduced-coordinate engineering gates, and long coupled runs remain open. Failed build/fixture trials are retained with successful repairs.

Build134 dedicated cache-only Human body time is 23.181222875 s versus 17.362774375 s control, with acceleration/anchor comparison errors 6.253%/6.118%. A matched tighter mechanical stopping pair takes 23.494940958/23.459087541 s; its 3.772%/3.520% errors still fail 1%. Neither test qualifies cache reuse or demonstrates a speedup.

Build137 finishes the opt-in ABI9 force stage's dependency tests: one GPU producer write yields the exact expected dt*M^-1*f response (error zero), explicit precommit rollback passes, and an actual GPU-blitted NaN fails without accepted publication. A parallel validation kernel checks producer and consumed force at the existing Stand status boundary before post-dynamics consumers; existing physical reconciliation owns restoration. The test distinguishes precommit abort from a committed GPU rejection. Existing candidate-FK and source-M callbacks now admit the explicitly versioned ABI9 tail with exact size checks; ABI<=8 remains unchanged. This is tested transaction plumbing, not completed implicit Human attachment coupling.
