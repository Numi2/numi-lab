# Prepare the 310 s native resting pair

This workflow prepares, but does not register or execute, one baseline/intervention
Numi Science v2 pair. It fixes the native outer step at 2 ms for 155,000 accepted
steps (310 s total), excludes the first 10 s, and observes the remaining 300 s.
The baseline and drive-reduction arms use the same 2 ms outer step and owner
assets; only the existing diaphragm/intercostal drive intervention changes.

## 1. Produce a genuine owner preflight

First resolve the final source and asset pins, then run a short baseline through
the existing Human owner command. This creates `invocation.json`,
`run-metadata.json`, and `native.log` from the actual owner launch. The study
preparer rejects hand-edited or incomplete receipts, a run that did not load the
pinned native runtime, an unverified terminal step count, and mismatched program
fingerprints.

The values below are the required accepted 2 ms path. Supply the final
owner-receipt paths; do not use a receipt from the 8 ms sensitivity run.

```sh
NUMI_LAB_ROOT=/Users/n/numi-human-performance-source-014 \
NUMI_BUILD_DIR=/Users/n/numi-human-performance-build-014 \
NUMI_HUMAN_ROOT=/Users/n/numi-human-resting-profile-001 \
NUMI_HUMAN_EXECUTION_STAGES=1 \
NUMI_HUMAN_TRAINING_PROFILE=1 \
NUMI_HUMAN_RESTING_TRANSACTION_PROBE=1 \
NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT=1 \
NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS=8 \
NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT=0 \
NUMI_HUMAN_STAND_SPARSE_OPERATOR=1 \
NUMI_HUMAN_STAND_CACHE_LIMIT_EQUALITY=1 \
NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES=1 \
NUMI_HUMAN_STAND_REDUCED_CHOLESKY=1 \
NUMI_HUMAN_STAND_REDUCED_BASE_PROJECTION=1 \
NUMI_HUMAN_STAND_DEFER_EQUALITY_DATA=1 \
NUMI_HUMAN_STAND_DEFER_EQUALITY_DIAGNOSTICS=0 \
NUMI_HUMAN_STAND_EQUALITY_DEFERRAL_FAULT=none \
NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM=1 \
NUMI_HUMAN_RESTING_ELIMINATE_FIXED_BOUNDS=1 \
NUMI_HUMAN_RESTING_CONSOLIDATE_LINEAR_BOUNDS=1 \
NUMI_HUMAN_SUPPORT_ANCESTRY_PRUNE=1 \
NUMI_HUMAN_PARALLEL_RESPIRATORY_MUSCLES=1 \
NUMI_HUMAN_PARALLEL_RESPIRATORY_GAS=0 \
NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING=1 \
NUMI_HUMAN_RESPIRATORY_SUBCYCLING=1 \
NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS=0 \
NUMI_HUMAN_STAND_CACHE_CONTACT_JACOBIAN=0 \
NUMI_HUMAN_STAND_CONTACT_WARMSTART=0 \
NUMI_HUMAN_STAND_FIRST_SIMD_CONTACT_SWEEP=0 \
NUMI_HUMAN_STAND_FUSE_CANONICAL_BODY_PROBES=0 \
NUMI_HUMAN_STAND_HYBRID_EQUALITY_FACTOR_CACHE=0 \
NUMI_HUMAN_STAND_ONE_PASS_ORDERED_LIMITS=0 \
NUMI_HUMAN_STAND_PGS_VELOCITY_RESIDUAL_TOLERANCE=0.0 \
NUMI_HUMAN_STAND_SPECULATIVE_CONTACT_DISTANCE_M=0.0 \
NUMI_HUMAN_STAND_ZERO_CONTACT_RESPONSE_FASTPATH=0 \
NUMI_HUMAN_SKIN_INFLUENCE_TILE32=0 \
NUMI_HUMAN_GPU_TIMING=0 \
NUMI_HUMAN_SUPPORT_GPU_TIMING=0 \
NUMI_MATTER_GPU_TIMING=0 \
NUMI_MATTER_GPU_TIMING_DENSE45=0 \
tools/numi human-resting \
  --body-scene <final-body-scene.json> \
  --anatomy-receipt <final-anatomy-receipt.json> \
  --tendon <final-tendon-payload.nhtendon> \
  --lab /Users/n/numi-human-performance-source-014 \
  --build /Users/n/numi-human-performance-build-014 \
  --output <new-2ms-owner-preflight-directory> \
  --seconds 2 --dt 0.002
```

The command must run from the Numi Lab checkout that contains the `numi`
dispatcher, and `NUMI_HUMAN_ROOT` must name the checkout containing the
updated owner launcher. This is a real 2 s GPU preflight, not a long-run result.
The owner runner records the effective settings in its invocation receipt.

## 2. Prepare and review the fixed v2 plan

Use final build/source inventories and actual owner-reported world and program
fingerprints. The control fingerprint must match the completed preflight log;
the intervention fingerprint must come from the native owner for the declared
drive change. Never infer or edit these identities.

```sh
python3 matter/tools/resting_intervention_study.py prepare-native-310s \
  --repository /Users/n/numi-human-performance-source-014 \
  --directory <new-plan-draft-directory> \
  --invocation <new-2ms-owner-preflight-directory>/invocation.json \
  --source-hashes <final-compiled-source-hashes.json> \
  --source-revisions <final-owner-revisions.json> \
  --parser-fixture <retained-native-csv-fixture.csv> \
  --world-fingerprint <native-world-fingerprint> \
  --control-program-fingerprint <preflight-control-program-fingerprint> \
  --treatment-program-fingerprint <owner-reported-intervention-program-fingerprint>
```

The command writes the existing `numi.science.plan.v2` draft, native identity,
model, and parser calibration. It refuses a different physical timestep,
unverified owner run, changed selected GPU path, enabled exploratory solver
route, or stale audit-reference pin. It never registers or launches a trial.
Review all generated inputs and limitations, then use the standard Numi Lab
owner CLI:

```sh
tools/numi science register <new-plan-draft-directory>/plan.json <study-directory-outside-checkout>
tools/numi science run <study-directory-outside-checkout>
```

The registered pair runs control then intervention. Review each native receipt
before the next declared arm and analyze only after both are retained.

## Audit cadence and limits

The plan pins reference 752 and the full-q 2 ms reference 801. Reference 752
supports the eight-root COM observer schedule and its retained physiology,
surface, and aggregate contact diagnostics for its 2 s, 8 ms condition. It does
not establish 8 ms temporal accuracy. Reference 801 retained 1,000 full-q
integration audit rows at 2 ms and 33 surface frames over 2 s. It is a bounded
correctness check, not long-horizon endurance.

In the 310 s pair, body/circulation/controller outer execution remains one
accepted 2 ms root at a time. The presentation cadence is 64 ms; the accepted
COM momentum observer groups eight accepted roots (16 ms); the full q
integration audit is disabled for the long trace. These are observation
settings. They do not coarsen, skip, or batch physical integration. The
full-q 2 ms reference and the segment-8 scheduling reference are separate pins
with separate claims.

The 64 ms presentation schedule still audits each displayed accepted surface
state. The native verifier requires full visible-surface skin/bed and finiteness
checks, whole-mesh triangle-area counts, functional-geometry status, and
functional-volume consistency for the displayed state. It records the actual
triangle count from the pinned anatomy rather than assuming a fixed mesh size.
This is a presentation-state geometry audit; it does not substitute for the
disabled per-root q-integration trace, and it does not establish tissue-interface
validity.

The plan remains a single deterministic simulation pair. It does not qualify
whole-body anatomy, validate a population or patient response, or establish
physiological or clinical validity. Each native arm still retains accepted
physiology, the complete visible-surface audit, and the transaction/rejection
probe.

The existing observation adapter records complete-breath windows, cumulative
aortic/pulmonary ejection, and complete filling/ejection-cycle counts. The
exploratory plan does not require a minimum count of complete breaths or
repeated cardiac ejections in its v2 validity gates. Inspect these retained
values before making any physiological claim; the short full-q reference
contains no long-horizon cycle evidence.
