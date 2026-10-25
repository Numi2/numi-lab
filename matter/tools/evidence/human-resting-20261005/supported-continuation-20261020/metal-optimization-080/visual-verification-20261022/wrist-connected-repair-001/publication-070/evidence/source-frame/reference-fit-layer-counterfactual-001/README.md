# 294 whole-layer reference-fit counterfactual

This is one source-only counterfactual. It computes each of 2,411 current 294 detailed-surface targets from exact D36 ancestry: 2,332 direct vertex IDs and 79 source-face barycentric rows. It checks all 61,950 native current positions against the full weighted 86-binding LBS reconstruction (maximum error 2.252 um), then applies the pinned D36-to-AP rigid placement and checks the original O/I/D layer placement against the AP point array (maximum error 0.124 um).

The per-column correction is target_AP minus original_AP_O. The same correction is applied to the corresponding O, I, and D nodes. This preserves each column's skin and fat thickness by construction, while testing the effect on the full 28,632-tet mesh.

## Result

The counterfactual is invalid. It inverts 457 tets: 206 in material 1 and 251 in material 2, in both double and Float32-rounded geometry. No baseline tet was nonpositive. Minimum mean-ratio quality falls from 0.002069 to 0.000427 (Float32-rounded: 0.000431); median quality changes 0.24587 to 0.24257. Material 1 signed volume changes 0.003452571 to 0.003454405 m^3, while absolute volume rises to 0.003604893 m^3 because of inverted cells. Material 2 signed volume changes 0.011652343 to 0.011658970 m^3, while absolute volume rises to 0.011726593 m^3.

Column thickness does not change in double precision: skin median 2.450779 mm (minimum 0.500226 mm), fat median 9.818526 mm (minimum 0.505074 mm). The requested source correction itself has median magnitude 0.0000672 mm, p95 1.26353 mm, p99 2.93115 mm, and maximum 11.5980 mm; the large residuals are concentrated in the 79 barycentric anchors (their p95 is 6.93913 mm).

This proves that simply copying the same O correction into I and D is not a valid connected-volume repair. It is not a harmonic fit, does not change NHSKIN, and has no bone/tissue, mass-remap, FEM, native-runtime, or physical admission result.

## Reproduction

On SSH Numi:

    cd /Users/n/Work/294-source-frame-audit-001/reference-fit-layer-counterfactual-001
    /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_reference_fit_layer_counterfactual
    /Users/n/numi-human-prep-venv-20261005/bin/python reference_fit_layer_counterfactual.py --output /tmp/reference-fit-layer-run

The test fixture checks direct/barycentric ancestry mapping, a regular-tet mean-ratio value, and preservation of column thickness under equal O/I/D displacement.

Frozen pins:
- builder SHA-256 214c6e8ae16c0a58fa9665ebbe1891d05fff6577c56083f49096826477b23861
- test SHA-256 4980ba5b4646603d2b910a561599987616b01026be3f8b13a118f9e2bf1f1466
- run report SHA-256 4a0a121c5c8f4ddd6a83af8666dbc81373b833e8f1b93db9523035c84da3f4d7
- output NPZ SHA-256 ac66cfd7a1f3139e8f63030c4889fb83000dab701700dc120bc3bbc69f6867d0