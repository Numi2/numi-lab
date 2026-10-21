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
