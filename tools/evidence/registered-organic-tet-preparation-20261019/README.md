# Bilateral optic-nerve offline organic FEM preparation (242)

This isolated CPU candidate tests an adapter into Matter's existing WorldSource/ObjectSource and Human mass-donation structures. It introduces no asset format, runtime solver, or new constitutive law. The adapter is not wired into the active NHA scene: neither nerve has a runtime body-source attachment, mass donor, deformation-owner mapping, or native admission receipt. The 15,098-tetrahedron bilateral geometry is offline preparation evidence only.

## Repeatable CPU check

From this directory on numi:

```sh
./tests/run_tetgen_organic_fem_cpu.sh
```

The runner requires the frozen Matter base `3059af430d0feaa7ce4af3c59fbc1ba9b04a214c` as an ancestor, verifies both canonical `W/apps` tissue headers byte-for-byte and by SHA-256 against the bundle, and compiles only through canonical `W/apps` (there is no bundled-header fallback). It runs no GPU work and does not invoke TetGen. Before these helper headers are materialized in `W/apps`, this published runner intentionally fails closed; the pre-publication isolated CPU receipt is retained in `evidence/CPU_TEST_RECEIPT.txt`. `NUMI_W_ROOT` may point to a different descendant checkout only if the same exact headers are present. The test executes the actual retained bilateral meshes and checks:
- FJ1772 / FMA50878: 1,728 nodes, 7,675 tets, 2,376 boundary faces, volume 7.4928696e-7 m3.
- FJ1819 / FMA50875: 1,701 nodes, 7,423 tets, 2,378 boundary faces, volume 7.4887870e-7 m3.
- 1000 kg/m3 reference density yields mass residuals below 2.2e-19 kg in the fixture.
- inverted tet, duplicate tet, Float32 collapse, and same-side shared-face inputs reject.

The files in `assets/` are the small exact outputs consumed by the test, not the TetGen executable. The independent geometry-preparation receipt is in `evidence/optic-nerve-tetgen-preparation.json`; source transform and seam evidence is in `evidence/optic-nerve-seam-audit.json`. Their source identities and command provenance are in `provenance/SOURCE_AND_COMMANDS.md`.

## Scope and limitations

The material in `source/matter/materials/human_optic_nerve_longitudinal_low_strain_isotropic_effective.nmatter` is an explicitly inferred homogeneous isotropic neo-Hookean proxy. It maps the reported preconditioned human optic-nerve longitudinal tangent modulus at 3% strain, E3% = 3.4 +/- 2.2 MPa, to an effective shear modulus using nu = 0.49. This does not fit the measured nonlinear curve, identify transverse behavior, represent viscoelasticity or fiber anisotropy, or calibrate this anatomy or person. Density 1000 kg/m3 is a reference mass assumption, not a source measurement. The test fixture's identity coordinates are not a runtime registration.

No all-body nerve integration, anatomical attachments, donor-mass transaction, external contact acceptance, or deformation response is claimed. Do not add the geometry to a runtime Human scene from this bundle alone.
