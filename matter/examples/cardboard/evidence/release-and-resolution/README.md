# Native release, tooling, and resolution evidence

This package advances the explicit liner/flute/glue instrument. It does **not**
qualify physical creasing, folding, manufacture, or box assembly. It includes
failed predictions and failed native trajectories, not just passing checks.

## Question, instruments, and predictions

`release-resolution-plan.json` was sealed before seven native trajectories. It
predicted a clean right-grip release with retained plastic state, and no more
than 10% change in phase-end paper plastic-strain RMS and grip moment under
mesh/time refinement. It binds source, material, executable and Metal-library
hashes. The right grip releases in both independently simulated environments;
the left grip remains fixed. Accepted release changes mass/constraint state
transactionally, preserves inertial momentum, and does not reset plasticity.

`force-tooling-plan.json` was sealed after the resolution failures and before
four follow-up runs. It asks whether a tighter residual threshold improves
force accuracy and whether a prescribed rounded punch can contact the layered
board without prescribing indentation to its nodes. These are deterministic
trajectories; frames are not independent experimental samples. No new physical
data were collected. Material and process references are published sources.

## Results

| Condition | Native result | Interpretation |
|---|---|---|
| Plastic right-grip release | 112 accepted steps; right reaction exactly zero during release | Constraint release and irreversible material history execute |
| Matching elastic release | 112 accepted steps | Dynamic springback control, not a permanent-fold benchmark |
| Resolution baseline | 144 accepted | Reference numerical trajectory |
| Half timestep, equal physical duration | 288 accepted | Phase-end moment changes 19.1–24.3%; 10% screen fails |
| Twice the length subdivisions | 144 accepted | Moment changes 13.0–27.8%; screen fails |
| Twice the width subdivisions | 144 accepted | Both observables within 10% at all four phase ends; narrow screen passes |
| Twice the thickness subdivisions | 112 accepted; attempt 113 rejected, status 10 | Endpoint comparison inconclusive; no substitute endpoint |
| Instrumented baseline, load/hold only | 48 accepted | Free-node force-imbalance L2 reaches 2.83848 N; reference reaches 0.08934 N |
| Tolerance `1e-6`, Newton 28, FGMRES 64 | First step rejected in both environments | Tightening alone does not resolve the numerical gap |
| Same tight budget, half timestep | First step rejected in both environments | No accepted paired endpoint for the revised sensitivity hypothesis |
| Initial rounded-punch layered-board trial | 17 accepted; attempt 18 rejected, status 6 | Wrong postCommit pose allowed an earlier endpoint overlap; retained as failed evidence |
| Corrected native endpoint, 20 µm travel | 16 accepted; attempt 17 rejected, status 6 | Native endpoint certification now rejects and rolls back the overlap |
| Corrected native endpoint, 10 µm travel | All 96 accepted | Complete shallow contact/hold/withdraw/settle cycle; not crease formation |

Plastic release ends at an end-section angle of 0.08964° versus 0.01248° for the
elastic control. Plastic maximum free speed is still 0.01857 m/s versus
0.0005511 m/s in the control; kinetic energies are 3.37e-9 and 6.88e-12 J.
This is a **transient**, not an equilibrium springback angle. The plastic run
retains maximum plastic-strain component 0.007729, while its unloaded reference
has zero plastic strain. `release-comparison.png` plots measured exported angles
and velocities, without modifying geometry. Its JSON binds the input CSVs.

The thickness rejection retains residual 1.24147e-4 against the unchanged 1e-4
threshold. Its raw certificate acceptance bit remains one; computed
`step_accepted` is zero. Both observations are preserved. The rejected-step OBJ
is the rolled-back accepted geometry, not geometry at the attempted new load.

## Force and constitutive diagnosis

The new readback copies the final unpreconditioned FEM residual in the same
command buffer, after preDynamics and before postCommit and filters the accepted free-node mask. For this non-mixed,
single-microstep coupon, its free-node xyz rows divided by dt measure force minus
inertia, including contact. They are not grip forces or a static equilibrium
certificate. Failed-row values are diagnostic only. The global certificate has
a normalization floor of one; an absolute 1e-4 impulse threshold allows different
force bounds at different timesteps. Direct instrumented measurements motivate
this hypothesis, but tighter runs failed and therefore **do not establish that
this is the sole cause** of the time sensitivity.

The independent FP64 material-point oracle in `constitutive-path/` found no
explicit timestep/rate term in the law. Selected 64-to-128 subdivision paths
changed plastic strain by at most 0.1125%, stress 0.0787%, and accumulated
multiplier 0.1776%. Those local checks do not reproduce the board's deformation
path, contact, global Newton solve or FP32 arithmetic; they cannot explain away
its larger sensitivity.

## Tooling failure and admission guard

A native capsule/finite-box fixture passes on a synthetic elastic pad: the
moving punch transfers impulse and moves the pad, while the separated stationary
control remains effectively still. Zero-radius boxes now validate using their
positive half-extents; malformed boxes and zero-radius capsules are rejected.
The layered coupon uses exposed boundary-face nodes for rigid contact and native
kinematic body records with consistent starting pose and velocity.

That fixture is insufficient to qualify board creasing. At layered-board step
16 (zero-based), the native solve accepted but the independently evaluated
commanded end-pose gap was -0.43168 µm. The next root began with the tool advanced
into that gap and rejected. Native projection leaves a non-dynamic prescribed
proxy at its starting geometry during the root. The initial caller also reused
that start pose for postCommit, so it certified the wrong realized geometry.
The corrected caller publishes the end pose to the existing native postCommit
contact certificate; overlapping endpoints now cause native rejection and FEM
rollback. Speculative velocity admission in preDynamics alone still does not
solve the endpoint contact equilibrium. Increasing slop or accepting penetration
would not repair that remaining limitation.

The probe now stops when an independently audited commanded endpoint penetrates
beyond a small FP32 coordinate-roundoff allowance; this is separate from native
step acceptance. `endpose-guard-plan.json` seals a regression on the known failing
trajectory. The result preserves the actual native accepted count and reports
the tool protocol as failed. The caller now provides consistent prescribed end geometry for native
certification. A successful score still needs feasible continuation/CCD and
contact equilibrium at that end geometry before
attempting a larger indentation or a 90° fold. `native-endpose-plan.json` seals
the corrected-endpoint regression and a new 10 µm shallow cycle. The corrected
20 µm rejection restores the previous accepted board geometry byte for byte.
A first endpoint-rejection fixture jumped 2 mm completely past its authored
top-face contact nodes and therefore did not trigger endpoint rejection. That
failed check is retained in `ctest-final-001.log`; it cannot test swept CCD. The
corrected fixture places the nose inside the top collision surface and checks
native rejection plus rollback. Continuous through-motion collision remains open.
The 10 µm cycle accepts all 96 steps; its smaller travel is an instrument
condition, not evidence that the failed deeper score was repaired. See
`tooling-analysis/` for certificate-checked contact and endpoint measurements.
That shallow cycle records contacts on 28 of 96 steps, peak punch reaction
0.28843 N, minimum sampled end gap 0.18727 µm, and zero punch contacts/force in
the stationary reference. These are native instrument measurements, not
experimentally validated contact forces.

## Evidence identity and reproduction

`runs/` contains every declared attempt's manifest, result, CSV, invocation and
receipt. Meshes, phase-end material states, phase-end geometry and failed-step
geometry are gzip-compressed. Decompress copies into a temporary directory to
rerun the analyzers. `source-snapshots/` deduplicates the exact sources by SHA-256;
each receipt maps its snapshot archive. Binaries and Metal libraries are bound
by hashes and remain in the local checkout. The full raw trajectories remain at
`/Users/home/cardboard-gap-evidence-20261006`. All native receipts recorded their
bound inputs unchanged across execution.

The original release/resolution runs predate the force columns and tooling
extension. Their reports use the current analyzer against the original,
receipt-bound exports; they are not evidence from the later probe binary.
`analysis/` records those version distinctions. The runner's diff captures tracked
changes and its source archives capture newly introduced helper inputs. This
package's checksum inventory additionally binds the published analysis and tests.

## Remaining qualification gates

| Gate | Present evidence | What remains |
|---|---|---|
| Manufactured component structure | Explicit orthotropic paper, sinusoidal medium, finite glue and shared interfaces | Grade-matched geometry/density/glue inputs; process residual state |
| Native release | Momentum, inverse mass, idempotence, restore and rejection rollback checked | Equilibrated released fold and resolution qualification |
| Numerical accuracy | Complete sensitivity screen, physical-unit residual telemetry, retained failures | Stable tighter force solve and time/mesh convergence |
| Tool contact | Native endpoint/rollback tests; complete 10 µm layered-board contact cycle; retained 20 µm failure | Contact equilibrium at prescribed endpoints, CCD, full accepted score cycle |
| Paper constitutive law | CPU/Metal checks of ideal Hill plasticity and finite-strain memory | Asymmetric tension/compression, hardening, damage, fracture and unload/reload calibration |
| Glue | Finite elastic bridges | Cohesive failure strengths, fracture energies, curing and moisture response |
| Crease/fold comparison | Published protocols and explicit unit/source conflicts | Source-matched board/tool recipe; attributable curve data and uncertainty |
| Full box assembly | No claimed demonstration | Blanks, multiple scores, cutting, flap self-contact, closure joints and held-out assembly tests |
| Manufacturing from fibers/webs | Authored finished-board initial condition | Forming, tension, heat, adhesive application, drying/cure and residual stresses |

The source gaps are detailed in [the crease/fold reference map](../../crease-fold-gap-priorities.md).
No ad hoc weak hinge, fitted visual pose, or unsupported glue failure parameter
was substituted for those missing mechanics.
