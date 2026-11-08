# Analytic sphere–box CCD distance and remaining convergence failure

Source change `941261f4d8b5e9b6e7c4a56bca456362d6e6cd2f` makes the shared
`ccdDistanceAtAlpha` use the existing analytic sphere–box witness in either
collider order. It changes no tolerances, iteration budgets, learner settings,
actions, observations or rewards. Full Release integration build passes.

## Independent distance qualification

Four prospectively seeded streams each contain16,384 queries:8,192 geometric
configurations and their swapped-order twins. They cover tiny face/edge/corner
gaps and overlap, enclosed spheres, thin crow boxes, random orientations,
interpolated sweeps and quaternion-sign equivalence. The standalone native
probe calls the actual production query; an independent FP64 NumPy OBB oracle
uses the actual FP32 inputs. Required distance and surface-witness tolerance
is8µm; normal and order-symmetry tolerance is1e-4. Exact repeat is also required.

The old query reports success but has up to25.98mm distance error, with
order-dependent separation and witnesses. The corrected candidate, fresh
confirmation and actual CMake-built admission streams all pass. Their maximum
distance errors are0.107,0.105 and0.134µm respectively; swapped normals and
separations agree exactly. The comparison is a distance-query qualification,
not a full trajectory or biological qualification.

The first witness reader reversed sphere/box roles. Original scripts, analyses
and failed labels are preserved. `analysis-revision.json` binds the unchanged
native data and original analysis; `analysis-v2.json` applies the corrected
reader. Four hand-solved cases and a deliberately wrong-order negative control
calibrate that correction. No native execution was repeated for reanalysis.
Fresh confirmation and build admission use the corrected reader prospectively.

## Native contact controls and generic regression

The crow fixture accepts7,296 transitions across positive, negative, neutral,
no-contact, chunk-partition and exact-restore controls. Physical contact/miss,
FP64 reward, actor truth exclusion and exact world/retina continuation pass;
there are no rejected physics steps in these controls.

The existing generic MetalWorld executable exits0 and reports
`metal_world_contact=ok` and `release_gate_40k=pass` after its physical, CCD,
capacity, transactional and replay assertions. Its measured wall throughput
is101,721steps/s in that separate benchmark. A helper initially checked for
the wrong string, `metal_world_contact=passed`; its unchanged native receipt
and output are reanalyzed without rerunning. This generic benchmark is not
crow training throughput or evidence of a matched performance improvement.

## Fresh learner trial: runtime prediction contradicted

The changed shader cannot restore an old shader-bound life checkpoint.
The fresh run recovers the original unbiased revision1 policy from Git and
uses the original development seeds,128lanes,160ticks,32physics substeps,
fixed LR.0003, gamma.999, lambda1 and two full-batch PPO steps. It was declared
to run20updates, with coupled checkpoints10/20, followed by64unused held-out
placements. Both>=20mm paired mean goal-distance improvement and>=20percentage
point goal-success improvement were required; adaptive LR and checkpoint
selection were excluded.

The native process fails after13completed learner updates (saved revision14),
388.954seconds into the command. Last complete coupled checkpoint is update10,
revision11,1,600accepted ticks per lane. No final trainer JSON or complete
accepted-lifetime ledger was published; partial artifacts are not a resumable
life boundary or a throughput qualification. No held-out evaluation ran.
The runtime-repair prediction is contradicted; behavior is inconclusive.

The retained error is `MR_STEP_DID_NOT_CONVERGE` in lane26, submission-local
control5, substep14, compiled pair100. Zero-step compilation identifies the
pair as `crow_head/collision` box against `crow_manipulandum/collision` sphere,
in the same109-pair physical world. Required pairs/raw contacts/manifolds/
constraints are5/12/5/12, below capacities109/256/109/160. This is a separate
pair from the prior lower-beak106failure. A correct signed-distance query
does not establish correct conservative advancement or event scheduling;
the exact failed projected sweep/event still needs instrumentation.

## Recovery and scope

`manifest.json` stores compressed, content-deduplicated evidence in actual Git
objects. `runtime-recovery.json` binds exact executables, library, shaders,
Python/source/assets and coupled checkpoint10. `restore_runtime.py` reconstructs
those bytes from Git. `verify_failure_recovery.py` requires the four query
streams, contact controls and exact failure/partial model state to reproduce
on the same device/OS, before hash/mode/mtime/closed-handle checked cleanup.
The verification receipt records the observed result separately.

No policy promotion, adaptive-rate trial, learned manipulation or biological
qualification follows from this study. The protected feeding policy remains
unchanged. Next work is a minimal native failed-sweep capture and an independent
conservative-advancement/event regression. The broader lifelong environment
still requires physical food handling, ecological variation and social learning.

### Observed recovery result and narrower analysis

The strict all-artifact reproduction prediction fails on the incumbent file.
The worker writes a frozen pre-process baseline from its loaded learner: the
fresh run starts at revision1 and checkpoint continuation starts at revision11.
The incumbent is a separate evaluation artifact, unused by the optimizer;
no deployment selection runs here. Its resumed bytes exactly equal the source
checkpoint candidate, as the code requires. Original incumbent bytes are also
restored and verified from Git. No native execution is repeated for reanalysis.

`verification-v2.json` records four exact query streams, all six contact arms'
physical arrays/metrics/task streams, the exact native failure, three exact
canonical training outputs and all41model/Adam arrays and metadata. The
original failed plan and verification remain intact. The narrower result does
not claim every process-local artifact executes identically. All294,561,071B
of temporary recovery files were removed after unchanged hash/mode/mtime and
closed-handle checks.
