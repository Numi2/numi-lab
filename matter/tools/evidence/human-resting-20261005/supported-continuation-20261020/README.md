# Resting tissue and ligament continuation, SSH Numi

This continues the existing native Human/Matter/Metal scene. The source and
outer-skin repair remain intact. It is a development increment, not completion
of whole-body tissue contact or long-horizon physiology qualification.

## Implemented

- Bilateral inferred radial-head RCL regional fibers extend the existing
  registered passive catalog from 26 to 28 rows. Endpoint bodies remain
  independent of the elbow frame child. A three-body regression checks this.
  No added mass, blood state or parallel ligament solver.
- A global bounded correction improves the skin-inner and fat-deep reference
  mesh while leaving all 2,332 exterior points, donor frames, mechanical deltas,
  source vertex/face ancestry and prism connectivity unchanged. The full 27,714
  tets remain in the existing bonded skin/fat object.
- The worst reference edge-matrix condition number falls from about 1.39 million
  to 12,831. Maximum inferred boundary correction is 0.390 mm. These are mesh
  construction parameters, not measured-person anatomy.
- Float32 emitted prism preparation now rejects nonpositive or sliver
  tetrahedra with signed determinant / longest-edge cubed below 1e-5. This is an
  initialization geometry check; runtime physical Jacobian floors and solver
  tolerances are unchanged.
- Original mass debits remain 3.4525707761420956 kg skin and
  10.487108964465744 kg fat. The builder infers density from these debits and the
  corrected volumes within existing material bounds. Material ancestry and
  single force/state ownership remain with Matter.
- The 20 original skin/fat attachment source and donor identities are frozen,
  so repairing coordinates cannot silently select a different force constraint.
- The existing GPU FEM/body interface supports finite isotropic elastic
  attachments at those same sites, with equal/opposite force and body moment.
  The former hard ties remain selectable as a matched reference control.
  Stiffness uses existing adipose shear, local thickness and tributary area.
  This is an explicit, uncalibrated effective connective-interface assumption,
  not measured ligament or enthesis stiffness. See elastic-attachment-022/.
- Failure-only diagnostics distinguish rejected J from rolled-back accepted
  node coordinates. Transfer diagnostics retain the actual source, donor and
  face of the maximum projection distance. Old failure evidence remains retained.

## Verification and evidence

One entrypoint builds the native scene and existing force probe, then runs the
four preparation fixtures, both geometry audits and the Metal force/replay
probe. Use a fresh absolute output directory on SSH Numi:

```sh
sh tools/verify_human_supported_tissue.sh /absolute/fresh/output
# Add --native for the 32-step integrated scene, or --rollback for its
# two-step injected-rejection check.
```


- mesh-quality-audit/: independent before/after source preservation, six-order
  prism study, emitted Float32 quality floor and thin-shell/control checks.
- mesh-quality-016/: rejected large fit (inverted elements); never integrated.
- mesh-quality-017/: smaller positive fit; not installed directly because its
  minimum thickness needed a numerical margin and volume compensation.
- mesh-quality-018/: final coordinates, thickness/frame provenance, bounded
  volume compensation, fit/freeze scripts, geometry receipt and native trial
  declaration. Previous source coordinates remain in c296 evidence.
- All three 4,619-face boundary surfaces have zero self-crossings and zero
  outer/interface/deep pair crossings in the Float32 audit. The assembled
  exposed 9,418-face FEM boundary is closed, consistently oriented, and has zero
  self-crossings. This does not prove clearance against every anatomical organ.
- Full-size FromLayers assembly and the negative API regression pass; the
  latter rejects a positive-volume, 1 micrometre / 1 metre shell sliver.
- native-witness-015/: failure at requested step 14, with the prior first
  13 accepted rows reproduced exactly. The old fat-face 3947 reference sliver
  has source-face parent 95151.
- Native 019 rejected step 19 at a different, well-resolved foot element after
  18 accepted steps. The matched original-20-hard-tie control 021 reproduced
  the same failure; it was not caused by the two newly selected sites. These
  results motivate finite attachment mechanics, rather than another quality
  threshold change. See native-quality-019/ and attachment-control-021/.
- Trial 022 passed the hard-tie failure point and accepted 20 steps (2.5 ms).
  It then rejected a fascia contact: global object 9, dense contact slot 1975,
  status CONTACT_FAILURE. This is the per-feature Armijo all-trials-rejected
  branch; its diagnostic values are energy and directional derivative, not
  penetration depths. It does not by itself prove crossing.
  The skin/fat failure did not recur before that point. Full FEM endpoint
  extrema were unavailable because the requested horizon failed.
- The consolidated 023 verification, including the force probe and a two-step
  integrated downstream-rejection rollback/replay, passed. This is a short
  numerical regression and not steady supported-body qualification.
- Native trial 022 is declared in elastic-attachment-022/TRIAL.md. It is mechanics-only and 32 x 125 us,
  maximum 4 ms. It cannot establish sustained support, five-minute breathing,
  physiological plausibility, or real-time performance.
- CPU and native builds use the installed Numi M6 Metal toolchain. No visual
  inspection was performed.

## Targeted contact continuation 024

The sparse existing contact trace now supports an explicit microtick selector
without requiring a single-object accepted-snapshot capture. It leaves the
physical path unchanged when disabled. Verification 024 passed the same four
preparation fixtures, two geometry audits, Metal interface force/energy probe,
and two-step integrated rollback/retry.

Root 20 / microtick 2 / slot 1975 reproduced the 022 failure at Newton iteration
13. All first 20 accepted CSV rows are byte-identical. All twelve trial
endpoints were collision-certified over their full fraction, had finite barrier
energy, and failed the per-feature energy-decrease check. The selected feature
is edge-edge between global fascia object 9 and skin/fat object 15.
The trace is retained in contact-armijo-024; it is diagnostic evidence rather
than a physical correction or sustained-contact qualification.

Safe hardlink consolidation of completed byte-identical generated packs freed
about 18.5 GiB while preserving all 98 paths; storage-dedup-024.json records
hashes, inode mappings, and before/after free space.

## Edge-edge precision repair 025

The 024 witness independently identifies separately rounded world-space closest
points as a source of gap error. A shared compensated relative segment witness
now supplies separation and normal to both mapped/unmapped EE generation and
barrier energy. The established closest-segment parameters, collision
certification, materials, physical timestep and validity floors are retained.

The existing native contact geometry checker now covers 48 nearest-feature,
48 mollifier, and six captured/control EE cases. On the three captured
Float32 endpoint sets, repaired distance agrees with the independent Float64
oracle within 0.378 picometres; the legacy rounded-point control misses the
base gap by 14.096 nanometres. Both full/tiny captured trial gap changes have
the correct direction. This is a targeted numerical result, not a physiological
or clinical validation. The fixture setup failures are retained separately.

The consolidated verification-026 passed preparation, geometry, Metal interface,
the expanded contact checker and the two-step integrated rejection/rollback
retry. Verification-025 retains an earlier harness path error; it did not
produce an integrated run. A separate 32-step integrated confirmation is
predeclared in contact-precision-025/TRIAL.md; its scope is only 4 milliseconds.

That run rejected after 20 accepted steps (2.5 ms). The previous root20/tick2
EE contact now has positive bounds and 17 accepted traced alpha trials, with
no contact rejection. The next tick rejects in the existing vascular
line-search owner (local row23, three completed microsteps); the 32-step
horizon did not pass. Full evidence is in contact-precision-025/native-integrated.
The row identity must be mapped through stable identifiers: source flow_02
connects stable compartment2 (brachiocephalic arteries) to stable compartment3
(upper-body arteries). Source endpoint numbers are not zero-based indices.
A source-bound initialization audit also documents why the release is not
a solved supported equilibrium.

## Remaining scope

The elbow MCL/LUCL candidates did not clear the bounded straight-path source
geometry search; annular wrapping is absent. The 28-row catalog is not all
human ligaments. Spinal, wrist and hand ligament/capsule tissue is not covered
by this catalog. The 26 live fascia regions are not counted as ligaments, and
the optional rigid-digit owner is not passive hand tissue. The existing knee
and other independently owned constraints are not duplicated. See elbow/evidence.md for source bounds and literature
priors.

All 54,551 vertices referenced by skin triangles have accepted-state transfer
rows. The 7,399 remaining payload vertices are unreferenced by any triangle;
they are not missing anatomical patches. Active rows retain fitting error:
the maximum projection is about 42.6 mm and has an explicit source/donor/face
witness. A full supported-motion transfer qualification remains outstanding. Three-layer material-column self-contact is enabled,
not general full-surface skin self-contact. The coarse source exterior still
has 45 open edges; the bonded FEM exposed boundary is closed by source-derived
sidewalls. All anatomical boundaries and surrounding-target interfaces still
require supported-motion qualification. The initial pose is a transient release,
not a solved static equilibrium. A short failure-free trial does not prove
sustained bed contact; real nonzero contact impulses must be measured.

The 330-second baseline, respiratory intervention/recovery, continuous native
video and complete body-contact qualification remain outstanding. The full
27,714-tet workload is much slower than the earlier reduced scene; no old
0.15-0.17x rate is claimed for this workload.

## Resources

storage-dedup*.json record exact-size/SHA-256 consolidation of completed,
unopened immutable generated packs. Every path/content is retained by
hardlinks. Always create a fresh output directory for a new run.
