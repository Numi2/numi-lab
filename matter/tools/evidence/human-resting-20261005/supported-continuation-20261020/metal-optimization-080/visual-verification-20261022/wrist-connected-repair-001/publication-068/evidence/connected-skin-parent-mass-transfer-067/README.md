# Connected skin/fat parent-reference mass transfer, 067

This isolated Numi-only increment supplies conservative parent-cell mass transfer
for unchanged and conformingly refined material tetrahedra. It does not alter the
canonical checkout or claim that a candidate mesh is anatomically admitted.

## API

deriveIdentityParentMeasures builds identity ancestry when child and parent
tetrahedron rows, topology, and materials are unchanged.
deriveSingleParentRefinementMeasures builds one-parent rows for a conforming
refinement from original vertex identities and inserted-node parent-tet plus
barycentric ancestry. A shared edge or face node may name one incident source
parent: the helper reconstructs its reference pullback point and maps that point
into each adjacent child parent. It rejects children whose pullback vertices do
not lie inside that one parent.

For a general overlap or pullback CSR, assembleParentReferenceMassTransfer accepts
raw per-child/per-parent reference measures m_cp and computes:

- A_c = sum_p m_cp (child ancestry measure)
- f_cp = m_cp / A_c (the owner CSR row fraction)
- M_c = sum_p M_p * m_cp / V_p (transferred cell mass)
- rho_c = M_c / V_c (mass divided by emitted geometric reference volume)

It checks positive finite inputs, sorted unique rows, material compatibility,
per-child implied densities against supplied material bounds, parent volume and mass
column closure, and per-material/origin-donor mass closure. It does not normalize a
failed column or create proof that supplied measures are geometrically correct.

## Evidence and limits

The copied NumiHumanUnpairedSkinFatVolume.hpp is the exact 2973 owner snapshot.
The integration test feeds helper-generated CSR and child ancestry measures into
that owner, calls the frozen compileWorld, and verifies heterogeneous per-child
mass reaches compiled node masses and inverse masses while material default
densities remain unchanged. Synthetic material bounds are skin 9001100 kg/m^3
and fat 850950 kg/m^3.

Production use still requires authoritative original per-tet masses, parent
reference geometry, typed material/origin-donor rows, node parent/barycentric
ancestry, and an independently pinned overlap or reference-pullback receipt. The
single-parent splitter cannot certify cross-parent retriangulation; such cells
must provide a complete overlap measure CSR. Detailed source-face ancestry remains
separate from volume-cell ancestry. No actual 027 or newly reconstructed volume
candidate was mass-qualified by this synthetic fixture.
