# Supported skin/fat transfer and regional ligament forces

Implemented and tested on SSH Numi (Apple M6), source commit
`05ab251d95af13ac9017cadb0f8913160d4ecd11`.

- 26 inferred bilateral hip, shoulder and ankle reference fibers use the existing
  GPU tension-only ligament owner. Existing knee owners remain separate.
- The outer skin, inner skin and deep fat share one validated rigid placement of
  the admitted supported reference. Attachments, frozen repairs, thickness,
  topology and physical mass remain invariant. This does not admit arbitrary
  articulated poses.
- Native 009 passed four 125 microsecond steps: 0.5 ms simulated, 88.839 seconds
  elapsed, maximum ligament tension 0.251753 N, zero net ligament-force residual.
  The scene also exercised skin/fat FEM and contact for all 26 fascia regions.
- Native 010 passed the late-rejection/retry fixture: physical state, vascular
  state, passive history and Brain/respiration/presentation history rolled back.
- [Exact-pose audit](geometry/native-009-actual-pose-26.json): 26/26 passed, with
  all 185 registered bone surfaces tested. Open source surfaces do not establish
  closed-solid containment or measured enthesis anatomy.

**Qualification limit:** these are numerical smoke tests, not sustained supported
motion. The initial support impulse is large; settling, recovery, arbitrary-pose
transfer, complete human ligament coverage and physiological/clinical validity
remain unqualified. Full expanded FEM throughput is far below real time.
No visual verification was performed.

[Run 009 invocation](native-supported-ligaments26-009/invocation.json),
[run 010 invocation](native-supported-ligaments26-rollback-010/invocation.json),
[focused CPU checks](cpu-checks.json), [26-row fixture](registered-ligaments-cpu.json)
and [quantitative summary](supported-skin-ligaments-20261020.json) are retained here.

From the repository root, after building the recorded native target and loading
the recorded Metal toolchain environment:

```sh
python3 matter/tools/evidence/human-resting-20261005/supported-skin-ligaments-20261020/reproduction.py \
  matter/tools/evidence/human-resting-20261005/supported-skin-ligaments-20261020/native-supported-ligaments26-009/invocation.json \
  build-integrated-validation-20261019/supported-skin-ligaments-20261020/replay-new-009 --steps 4
```

Use a fresh destination. For rollback, select run 010's invocation and use
`--steps 2 --rollback`. Invocation paths reference the retained SSH Numi assets.
Original 009/010 provenance bytes are preserved. Their `geometry_audit_sha256`
field names the historical source-default audit; it is not current-pose
admission evidence. Use the exact-pose audit linked above and native startup
admission instead. Future reproduction receipts name that audit explicitly.

## Historical diagnostic record

The following sections preserve the earlier failures and interim observations.
Later 009/010 results above supersede interim statements about unavailable native
catalog runs; failed or source-default audits are not promoted to current-pose
qualification.

# Supported skin, fat, fascia, and ligament evidence checkpoint

This compact receipt points to retained run artifacts under
`build-integrated-validation-20261019/supported-skin-ligaments-20261020/`.
It does not copy the large output packs, terminal states, source snapshots, or
logs. Earlier failures remain intact.

## Vascular line-search failure and fix

Runs 003–005 reproduce one coupled vascular failure on the same four-step,
125 µs supported-pose invocation. Run 003 failed on the fourth step after
three accepted steps, with Matter status 10 and diagnostics
`[0.000244140625, 6.0582912908557773e-08, 6.0596562434511725e-08, 12]`.
Run 004 added an owner witness: stable compartment ID 7,
`compartment_06_abdominal_aorta`, absolute-volume continuity row 6, normalized
tolerance `1e-5`; the final tested line-search alpha was `0.00048828125`, and
merit rose from `6.0582912908557773e-08` to `6.0596562434511725e-08`. Its worst
row was `21.35245` times tolerance.

Trace 005 shows the cause: coupled FGMRES stopped at its global relative forcing
estimate while vascular row 6 remained under-solved. The row residual changed
from `-2.1352185285650194e-4` to `-2.11363731068559e-4` (1.0108% reduction),
leaving it `21.1364` times its authored normalized tolerance. Arnoldi estimated
relative residual was `0.235`, just below the old target `0.25`; this is an
Arnoldi estimate, not a measured true residual. The evidence supports an
inexact-Krylov stopping cause, not the earlier FP32-cancellation hypothesis.

The narrow fix caps the vascular-coupled FGMRES inner target at the existing
authored normalized vascular tolerance. It leaves nonvascular relative
stopping and the outer equation certificate/tolerances unchanged. Run 006
replays the same native scene and executable with the corrected Matter shader:
exit 0, four accepted steps (`0.5000000237 ms` simulated), terminal FEM audits
passed, and no common-field failure receipt. Retain 003–005 as failed evidence;
006 is a short numerical replay, not a long-horizon or physiological
qualification.

## Focused checks and exact identity

The native FGMRES regression passed 24,833 checks. It rejects vascular Arnoldi
estimate `5e-5` and accepts `5e-6` against `1e-5`; it also checks the existing
FEM-only true-residual-kernel contract and explicitly does not claim a vascular
true-residual measurement or full-FEM-operator qualification. The passive
ligament owner check passed 18 cases on Apple M6 (`max_relative_force_error`
`9.28221724e-8`, compression zero, low-strain response nonzero, force/moment
closed). The updated skin rigid-placement test now recomputes the expected
current local point from the body pose and passed (binary SHA-256
`1be86b8c8bcde0e3a07c7564b28a8d093e1b5f9d4535be080234740ebe8bed50`,
source SHA-256
`c18063cf704db5bffd40ff6fb355ef5453cebdb4ddf520513c580f7807a7ce0f`).
Accepted-transfer and mixed-transfer CPU checks also passed; exact hashes and
outputs are in `cpu-checks.json`.

Run 006 provenance records source base `6058573c89013488764f78a220350d4ca5efc407`,
executable SHA-256 `ef075f2897920b41f8b232314477bdba0fc95124122cb1557f5d84cdf62389a1`,
and corrected `NumiMatter.metallib` SHA-256
`869fdc8828a35dc055b680d0dade3c7627dcb7037e18a5d6eea5cbbbe1fc9090`. The
runtime reported device model `Apple M6`; the JSON stores a hash of that model
label only, not a unique hardware identifier. The executable captured after combined build 008 (SHA-256
`9fa577759e616b6b4e2b43cb24cf9062b1a938dfd728858a2662e52919b5fea4`); that
intermediate binary had no completed native run at that historical checkpoint; final 009/010 use the later source and binary recorded below.

## What the native replay exercised

The 006 invocation is `--mechanics-only`, with one supine flat-bed supported
reference pose, a root-pose transform, two `pose-q` overrides, and four
125 µs steps. It ran whole-body skin/fat FEM (13,857 skin and 13,857 fat tets;
6,996 shared FEM nodes; 4,709 contact nodes) and registered reference-FEM
fascia (26 source regions, 12 cooked objects, 1,202 nodes, 2,127 tets, 495
body-contact samples). These are actual native mechanics executions for this
brief pose window. Geometry/material scope remains limited: skin/fat thickness
fields are constructed rather than subject-measured; fascia material is a
reference proxy, not clinically calibrated; fascia self-contact was disabled.
This is neither a recovery test nor sustained supported motion.

Run 006 did **not** exercise the new passive ligament force rows: the runtime
reported `passive_ligament_force_rows=0`. The two inferred ACL geometries were
admitted for reference-pose geometry only, with
`separate_OpenKnee_ACL_or_passive_row=disabled`. Runs 007 and 008 failed during registered-ligament catalog cooking at the
explicit q36/q74 shoulder pose, before mechanics. Run 009 below is a later
native replay using the revised catalog source; its short terminal audit records
26 registered passive force rows. This establishes force ownership only for
that run's current pose and 0.5 ms window.

The retained static/source-default geometry audit historically reported 20
side-specific inferred regional fibers with catalog-header SHA-256
`97eb263d79a3b811da1817e4d44a11ca002dcb13fea1b9a37e5a128904eb1595`.
That result is contradicted for the explicit runtime shoulder pose, not merely
stale by hash. Native attempt 007 used the exact same header hash and applied
`pose-q` overrides 36=0.10 and 74=0.10; catalog cooking failed on the
superior glenohumeral capsuloligament's side-0 endpoint-normal check. Diagnostic
attempt 008 reproduced both sides at those overrides: side 0 facing witness
`[0.636102, -0.011944]`, triangles `[1136, 645]`, surface residuals
`[0.694787, 0.000008]` mm; side 1 `[0.697598, -0.008429]`, triangles
`[24501, 2641]`, residuals `[0.975866, 0.000002]` mm. The static audit is
therefore retained only as historical source-default evidence. The explicit
upstream shoulder pose changes FK endpoint normals; matching the historical
header hash in 007 rules out stale hash as the sole cause.

The native owner constructs a runtime surface for every retained source bone
record and checks its inventory against the full 185-record payload before
cooking ligament rows. Runs 007/008 reached the row-specific SGHL normal check,
so that surface-count guard did not reject; it demonstrates full surface
inventory coverage, not ligament path clearance. Those two runs stop during
catalog cooking, before mechanics. Run 009 then admitted 26 current-pose rows
and exercised 26 passive force rows with the revised header. This is evidence
for that exact header and brief pose only; the verifier's pose/header receipt
remains pending, and no catalog completeness or sustained-physics
qualification is asserted. The source endpoints are
seed projections onto registered BodyParts3D bone meshes, not subject-calibrated
entheses. Cited literature supplies stiffness/area priors, including
cohort-derived dimensions, while effective area can be a stiffness-preserving
normalization. Construction uses zero prestrain and does not model ligament
failure. The CHL is omitted because mesh witnesses failed admission. Source
bone surfaces are open; triangle intersection and normal checks cannot prove
closed-volume containment or full path clearance. No physiological or clinical
validation follows from these checks.

## Native 26-row supported-pose run 009

Run 009 completed four 125 µs steps (0.5000000237 ms simulated; 88.839 s
process wall time) on the same supine flat-bed reference pose with root-pose
transform and q36=q74=0.10. The runtime admitted 26 registered ligament rows
against 185 retained current-pose bone surfaces under the all-current-pose
path guard. The terminal runtime had 26 passive force rows, zero legacy
reduced rows, and accepted the latest ligament transaction. Its terminal
maximum tension was 0.251752793789 N; effective stretch ranged from
0.999842166901 to 1.00025689602. Force residual was zero and moment residual
was 2.03643642419e-9 Nm.

The whole-body skin/fat path rigid-placed 2,332 vertices and ran 13,857 skin
plus 13,857 fat tetrahedra with 6,996 shared FEM nodes. The skin/fat and 26
regional fascia FEM terminal audits passed; the fascia had 12 objects, 1,202
nodes, 2,127 tets, and 495 body-contact samples. The FEM terminal audit passed
all four steps. These are short numerical execution checks only.

Run 009 is bound to executable SHA-256
`5a4cb6b812fddffd9a0e7317bff1a7fccf62c8eef864b7f0768360674cc0ed34`,
Matter metallib SHA-256
`869fdc8828a35dc055b680d0dade3c7627dcb7037e18a5d6eea5cbbbe1fc9090`,
registered-ligament header SHA-256
`b49373a98089c58311a87a9f7b199ae5d8ac3b00e6fc1e6ecc4da8936231190b`, and
native probe SHA-256
`f143e175643a5596afd4e37761e65c5ec85b25e5d48105e568883a2377a236b9`.
The run's helper-generated provenance retains
`geometry_audit_sha256=bfa030b36d1efa36ddff6f19889eb2d6c85a472de8f45d0985f5f28a19c975e6`;
that field identifies the historical source-default audit discussed above,
not the explicit-pose native admission. The current-pose runtime admission is
recorded directly in run 009 stderr. A separate verifier receipt is still
expected to bind its exact pose/header.

The geometry and material limits above still apply: skin/fat thickness is
constructed rather than subject-measured, fascia is a reference proxy, and
ligament anchors remain inferred seed projections rather than measured
entheses. This 0.5 ms replay does not establish a complete ligament catalog,
sustained supported motion, recovery behavior, or physiological/clinical
validity.

For a fresh four-step output directory from the repo root on `numi`, run:

```sh
python3 build-integrated-validation-20261019/supported-skin-ligaments-20261020/replay_invocation.py \
  build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-009/invocation.json \
  build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-rerun-011 --steps 4
```

The destination must not already exist. Before comparing the rerun to 009,
require the helper-generated executable and metallib hashes to equal the 009
hashes above; the helper records current hashes but does not enforce a match.

## Native shared-tissue late-reject rollback check 010

Run 010 used the same executable, metallib, catalog source, and supported pose
as 009, with a two-step invocation and the shared-tissue rollback fixture. It
injected GPU stand status 9 after dynamics at accepted-publication finalization
for reject step 1. The native witness reports the late-reject retry passed:
q/v/root/muscle state, Matter vascular state, passive adapter history, and
brain/respiration/presentation state remained unchanged; body and DOF
properties were restored; retry succeeded in the same context. The run then
completed two accepted 125 µs steps (0.2500000119 ms); FEM and fascia-FEM
terminal audits passed, with 26 passive force rows and zero legacy rows.

The rollback terminal record reports accepted passive-ligament state,
0.0806076079607 N maximum tension, stretch range
0.999948382378–1.00008225441, zero force residual, and
4.11747206396e-9 Nm moment residual. This checks rollback and retry on a short
window; it is not sustained supported-motion, complete-catalog, measured-
enthesis, or physiological/clinical qualification.

To replay the rollback invocation into a new output directory from the repo
root on `numi`, run:

```sh
python3 build-integrated-validation-20261019/supported-skin-ligaments-20261020/replay_invocation.py   build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-rollback-010/invocation.json   build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-rollback-rerun-012 --steps 2 --rollback
```

The destination must not already exist. Require the helper-generated executable
and metallib hashes to match the hashes recorded for 010 before comparing
results.

## Retained artifact pointers

- `build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-003/` — initial failure and original invocation.
- `.../native-supported-vascular-witness-004/` — row-owner/normalized-tolerance witness.
- `.../native-supported-vascular-trace-005/` — compact FGMRES trace.
- `.../native-supported-vascular-fix-006/` — passing short replay, provenance, regression receipt, and diagnosis JSON.
- `.../native-supported-ligaments-007/` and `.../native-ligament-diagnostics-008/` — q36/q74 runtime-pose catalog-cook failures and exact FK normal witnesses.
- `.../native-supported-ligaments26-009/` and `.../native-supported-ligaments26-rollback-010/` — compact invocation, provenance, execution, admission/terminal excerpts, and coupled traces; full native outputs remain under the build validation tree.
- `.../cpu-checks.json`, `.../ligament-precision-after.log` — focused transfer and constitutive checks.
- `tools/audit_final_registered_ligament_catalog.json`, `tools/audit_ligament_regional_paths.json` — static inferred geometry receipts; see their classifications and open-surface caveats.

Machine-readable quantitative summary: `supported-skin-ligaments-20261020.json`.
