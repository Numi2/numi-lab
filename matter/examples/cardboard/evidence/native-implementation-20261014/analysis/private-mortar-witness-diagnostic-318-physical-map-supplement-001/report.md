# Native 318 exact-map physical geometry supplement

This supplement refines (and does not alter) the exact planar map audit in the separate diagnostic-overlay package. Inputs are decoded from native 318 `three_patch_covered_crossing`, not rebuilt from source expressions. Source log SHA-256: `276aea2f30ce703d6308f4d82ef256a284081ca18e3873b1195ec1fe2bf2708c`.

The captured center weight in each face is `a = 11184811/33554432`; the raw xyz sum is `3a = 33554433/33554432 = 1 + 2^-25`. The contact partition consumes x/y and reconstructs z, yielding `z_contact = 1 - 2a = 11184810/33554432`; raw cohesive weights have `z_raw = 11184811/33554432`. Their weight-space difference is exactly `(0,0,2^-25)`.

For the uploaded left parent nodes (1,2,3), `P2=(0,L,c)` with `L=8589935/2147483648 m` and `c=13743895/68719476736 m`. Therefore the raw cohesive patch center differs from the contact-reconstructed center by exactly `2^-25 P2`:

- `Δx = 0`
- `Δy = 8589935/72057594037927936 m` (about `1.1920928e-10 m`)
- `Δz = 13743895/2305843009213693952 m` (about `5.960465e-12 m`)

Each side-A raw physical patch vertex exactly equals its corresponding side-B vertex for all three faces in the uploaded base geometry. The raw three-patch physical area sum is not exactly the flat parent area: using the captured full xyz weights and node positions, the decimal sqrt evaluation gives parent `8.000000759959238934293512812701010261662304401397705078125e-6 m²`, patch sum `8.0000007599592390408749193799572501591239930188833472614735401394508229802531980e-6 m²`, excess about `1.0658140656725624e-22 m²` (relative `1.3322674555320828e-17`). This is a minute off-simplex map/area discrepancy, not a side-A/side-B opening.

So the exact planar x/y maps have zero complement and zero positive-area overlap, while the authored full xyz map has a slight normalization/affine-consistency error. Do not fix either by dropping tiny partition cells or silently renormalizing authored weights. Keep it distinct from the yet-unknown CCD witness in the four native 318 rejected cases. The witness overlay remains diagnostic-only and has not been GPU-run.
