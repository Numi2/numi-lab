# Preregistered small-strain ophthalmic-strand screen

## Question and boundaries

Can a single effective linear axial relation fitted to one straight 8-0 Nylon specimen predict the force measurements of four other straight specimens over 2–4 mm extension? This is a simple material-data screen, not a simulation, identified constitutive law, knot validation, or clinical prediction. Machine compliance and grip effects are included in the observations. The study's initial gauge length is 120 mm and the imposed crosshead speed is 30 mm/min.

Source: Dalai et al., [PLOS ONE article and S4 workbook](https://doi.org/10.1371/journal.pone.0343682); supplement retrieved from the publisher, workbook SHA-256 `49a0f7f4875bc6354367c3efd832ec26642355965fe0cae31ffc950280045978`. Its “8-0 Nylon” sheet labels `8-0NS1` through `8-0NS5` as separate straight specimens. The sheet's first three header rows were inspected before this plan; no numerical rows from that sheet have been inspected. Another sheet's first few rows were inspected for the file schema and cannot be called fully unseen.

## Fixed protocol and prediction

- Development specimen: 8-0NS1. Held-out specimens: 8-0NS2, 8-0NS3, 8-0NS4, 8-0NS5.
- Use all raw paired extension/load observations with 2.0 ≤ extension ≤ 4.0 mm; retain negative force values, original order and duplicate extensions, without baseline adjustment, smoothing, reweighting or time alignment.
- Fit `F = a + b x` by ordinary least squares to the development specimen's selected samples, with x in mm and force in N. Require at least 20 finite pairs and a finite positive slope.
- On each held-out specimen use all selected raw samples; require at least 20 finite pairs and nonzero summed absolute force. Compute `sum(abs(F_pred - F_measured)) / sum(abs(F_measured))`.
- Prediction: each of the four held-out relative absolute errors is at most 10%. One failure, invalid input, or missing coverage fails the screen. Keep every per-specimen result. No adjustments to the interval, coefficients, or threshold after observation.
- Workbook hash, selected column headers/units, raw parse and exact program identity must be recorded. Test errors and fixture limitations remain evidence. If the screen fails, retain it and define a new model/prediction before using any fresh data; do not report it as successful calibration.

Passing would establish only the stated held-out comparison for this range and batch. It would not demonstrate Numi rod mechanics, rate dependence, bending, twist, damage, knot retention or tissue interaction. A separate native Numi fixture and its force/energy/replay gates would still be required.
