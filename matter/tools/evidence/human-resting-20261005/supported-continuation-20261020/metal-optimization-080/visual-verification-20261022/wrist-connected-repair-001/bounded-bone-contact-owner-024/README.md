# Bounded rigid-bone triangle contact owner

This increment adds immutable, prepared bone-triangle banks to the existing
Human FEM/body interaction owner. The GPU chooses the closest finite triangle
within each prepared bank, with a deterministic stable-ID tie break. Force,
reaction wrench, provisional validation and accepted audit use the same rule.
Leaving the declared distance bound rejects admission. This is not a global
bone broadphase, proof that a bank contains the globally nearest face, or an
anatomical bank producer. No new per-step CPU physics or contact history owner
is introduced. Legacy 64-byte contact and 96-byte dispatch records keep their
sizes; optional source spans change the C++ layout, requiring matched library
and caller rebuilds (load ABI14).

## Verification on SSH Numi / Apple M6

- CPU closest-feature, tangent, force/moment, invalid/capacity, duplicate-ID,
  and full-versus-filtered solver-range indexing tests pass.
- Final Metal coupon passes: nonzero contact, closest-face switching,
  empty/zero-range legacy behavior, radius rejection, and provisional-status
  rejection outside the bank. The initial final-coupon assertion failure was
  a test-only legacy stiffness-field mistake; its original result is retained.
- Fresh runtime, native Human viewer and existing tendon/FEM probe build pass.
  Exact bound dirty-source inputs and binary hashes are retained; this is not
  a clean-main whole-source build claim.
- Existing surface-attachment CPU/Metal/Runtime suite passes all nine stages.
- Extended actual adapter/Runtime fixture passes: inactive bank bindings,
  bank-source fingerprint separation, two accepted 0.1ms spring-driven steps,
  rejection without accepted-node/cursor/history changes, and bitwise replay.
  Its body is fixed and its bank does not contact; nonzero bank contact is
  exercised by the separate production-kernel coupon.
- Existing coupled tendon/FEM owner regression passes with legacy spans empty.

No new joint skin, bone bank, material property, ligament route or natural
range of motion is admitted by this increment. Elbow/shoulder folding,
left-ankle dorsiflexion crossings and the hand layer mapping remain separate
unfinished anatomical/physical qualification work. The viewer was built but
no visual inspection was performed for this increment.

All prior build021 binaries remain untouched for ABI13 replay. Only exact
byte-identical temporary compiler products were retired, after rehashing both
copies and checking for open consumers; the duplicate-retirement manifest
identifies each retained copy. Source, receipts and failed evidence remain.

The original patch and compiler log are retained as lossless gzip files;
lossless-evidence-manifest.json binds compressed and original hashes.
