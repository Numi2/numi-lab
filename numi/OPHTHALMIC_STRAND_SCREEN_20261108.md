# Ophthalmic strand held-out data screen

The [fixed screen](OPHTHALMIC_STRAND_PREDICTION_20261108.md), committed as `40921e133` before inspecting numerical values from the chosen 8-0 Nylon sheet, passed its predeclared 10% relative absolute force-error limit for all four held-out straight specimens. This is a measured-data comparison; no Numi simulation was run.

The development specimen's affine fit over 2–4 mm extension was:

```text
force_N = 0.04886680223398625 + 0.020256155067237984 * extension_mm
```

| Specimen | Raw samples in fixed interval | Relative absolute force error | Gate |
| --- | ---: | ---: | --- |
| 8-0NS2 | 400 | 3.0550% | pass |
| 8-0NS3 | 400 | 3.0172% | pass |
| 8-0NS4 | 401 | 1.7589% | pass |
| 8-0NS5 | 400 | 2.9593% | pass |

The development fit used 400 samples. These specimens come from the same reported material, gauge, experiment and loading rate; this is not a test across lots, loading rates or independent laboratories. The nonzero intercept is part of this restricted affine fit. It must not be silently treated as a zero-force rest state, as a material modulus, or as proof of preload. Extrapolation outside 2–4 mm is unqualified. Bending, twisting, knotting, fracture and tissue contact remain untested.

The publisher's S4 data workbook SHA-256 is `49a0f7f4875bc6354367c3efd832ec26642355965fe0cae31ffc950280045978`. The executed [screen program](measurements/screen_ophthalmic_strand.py) SHA-256 is `74d08021c14923c413d19d02ead47353834c094db5bcff7fcbc1808d3c686263`, committed before execution at `c4a8dd181`. It verifies and parses the same bytes, checks specimen names and units, rejects unpaired or nonfinite cells, and preserves raw samples; a synthetic exact affine case and a deliberate held-out failure were checked before execution. The complete machine-readable output is [retained](measurements/evidence/ophthalmic-strand-screen-001.json).

Reproduce on the original workbook:

```sh
curl -fsSL 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0343682.s004&type=supplementary' -o s4.xlsx
python3 numi/measurements/screen_ophthalmic_strand.py s4.xlsx --output screen.json
```

The next physical step is a native strand test whose initial state and effective axial response are defined consistently with the observed force–extension interval. It must compare against the unchanged held-out values, independently check reaction force and work, and exercise native save/restore. Until that passes, the data screen does not qualify Numi's rod implementation or the continuous surgical workflow.
