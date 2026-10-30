# Authenticated operational child faces in the detail preflight

The detailed-skin CSV loader previously accepted changed child-face rows only through the conforming 002->530->006 lineage. An independently retetrahedralized skin/fat volume uses the existing conservative spatial mass route and must not claim that lineage. The loader now recognizes its explicit child/parent face rows when the existing source-only preflight permission is present and the prepared mass-path hash still binds the current geometry, labels, operational domain and cached transport receipt. It rejects missing pins, stale geometry/labels/receipts and conflicting conforming or identity state.

The default runtime identity requirement and all numerical mass, material, donor, ancestry, projection, contact and physical-admission checks remain in place. This change prepares detail transfer for the repaired volume; it does not admit an incomplete Human mesh.

Assert-enabled and AddressSanitizer/UBSan detail-bridge tests pass, including reordered child/parent rows, canonical-map export/reload, stale input rejection and unchanged output after failures. The new synthetic case tests routing and proof binding; numeric mass transport is owned by the separate registered-import/mass suites. The exact main probe compiles and links with the modified header on SSH Numi. No Runtime or accepted physical step was created.
