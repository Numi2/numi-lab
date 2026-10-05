# Five-minute coupled physiology numerical endurance

This is a 300.000014 s run of the reduced coupled Matter/CVSim21, respiratory gas compartments, and NumiBrain chemoreflex instrument. It did **not** contain a qualified anatomical whole body or measured-person model; the runner explicitly reports `no_anatomy_or_resting_claim=true`. It started before the intervention study was registered, so it is retained as an exploratory numerical endurance run and is not a registered intervention control.

The run accepted all 150,000 steps at 2 ms on an Apple M4 Pro. Wall time was 1,093.0714 s, GPU time 1,073.1284 s, and achieved RTF 0.274456. It recorded 9,375 accepted-state rows. The original CSV is retained on the Mini at `/Users/n/numi-human-resting-evidence-20261005/dense45-endurance-001/baseline-300s.csv`; this repository holds its gzip copy and log.

The late baseline window, 120–300 s, had the following observed values:

| Output | Mean | Observed range |
| --- | ---: | ---: |
| PaCO₂ | 39.7335 mmHg | 39.5646–39.8458 mmHg |
| PaO₂ | 99.2362 mmHg | 98.9166–99.6989 mmHg |
| SaO₂ | 0.9717 | 0.9714–0.9720 |
| Tidal-volume state | 541.87 ml | 538.29–543.71 ml |
| LV pressure | 37.86 mmHg | 2.04–110.11 mmHg |
| Aortic pressure | 89.12 mmHg | 68.51–108.85 mmHg |
| Pulmonary-artery pressure | 14.77 mmHg | 9.24–22.03 mmHg |

Airflow integration over 120–300 s gives 6.143 L/min inspiratory minute ventilation. The breath counter advanced from 21 to 55 over that window, about 11.3 breaths/min. Across the full run, there were 350 complete filling/ejection cycles (about 70/min), final LV stroke volume was 70.04 ml, and cumulative aortic/pulmonary ejection was 24.521/24.512 L. These outputs show continuous reduced-model cycles and circulation; they do not establish anatomical myocardial contraction or clinical cardiovascular function.

The 21-compartment blood total ranged from 5,149.9992 to 5,150.0965 ml and ended at 5,150.0960 ml. The reported maximum absolute difference from the 5,150 ml initialization was 0.09685755 ml, about 1.88×10⁻⁵ of total blood volume. `circulation.w` is a running maximum of `abs(sum(vascular compartment volumes) - 0.00515f)`, not a per-step mass-equation residual. At the float32 reference value 0.00515 m³, one ULP is 0.0004656613 ml; the reported maximum is exactly 208 ULP. The current trace does not expose every 2 ms nonlinear mass-equation/Jacobian residual, so it cannot apportion accumulated offset between solver residual and float32 updates. No correction or rebalance was applied. The O₂ ledger error remained 0.00011642 ml STPD; the CO₂ ledger error reached 0.00046566 ml STPD. Cumulative aortic and pulmonary ejection differed by 9.4343 ml over the run.

For scale only, adult ABG references commonly place PaCO₂ at 35–45 mmHg and PaO₂ at 80–100 mmHg; this run's PaCO₂ stayed in that band, while the full-run PaO₂ mean was 100.34 mmHg and therefore slightly above that conventional interval. A published resting reference gives about 0.5 L tidal volume at 12 breaths/min (roughly 6 L/min); this run's late-window tidal-volume state and integrated ventilation are near that scale, with rate somewhat low. The alveolar ventilation relation predicts PaCO₂ to vary inversely with alveolar ventilation at fixed CO₂ production. These comparisons are plausibility checks against simplified reference values, not clinical validation; the modeled sampling and gas compartments are reduced-order.

References: [NCBI Bookshelf, adult ABG ranges](https://www.ncbi.nlm.nih.gov/books/NBK610832/table/ptviii.tab3/?report=objectonly); [review of resting breathing parameters](https://pmc.ncbi.nlm.nih.gov/articles/PMC8672270/); [NCBI Bookshelf, PaCO₂ and alveolar ventilation relationship](https://www.ncbi.nlm.nih.gov/books/NBK493167/).

Reproduction command on the Mini:

```text
/Users/n/numi-human-resting-build-20261005/matter/numi-human-resting /Users/n/numi-human-resting-lab-20261005/matter/tools/fixtures/cvsim21.native.v3.json /Users/n/numi-human-resting-lab-20261005/matter/examples/resting-reference-respiration.json OUTPUT.csv --steps 150000 --dt 0.002 --vascular-dense45
```

Source base at launch: `42fb4997a10661633db1ba8c2283bc4a3cc0450f`, with the exact instrument source hashes retained in the adjacent Dense45 pair receipt. Binary and loaded-library copies are retained under `/Users/n/numi-human-resting-evidence-20261005/dense45-endurance-001/frozen/`.

```text
binary SHA-256: f1121f70f0f3bb1aa4ca93559922e42488e8bb6df49d1c987a70d0e9d620ee81
NumiMatter.metallib SHA-256: 8c5f0c6078f7f385b3cce3b08b816df95721528cc797da27e134c14717e03a93
HumanRespiration.metallib SHA-256: 079714a5efe672ab5e5040814e434e996188c2eaeed0bffee8392cb558dc312b
CVSim21 input SHA-256: eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818
respiratory parameter input SHA-256: 27100bf8941fcd623bba1ea87474f6fe6a4b1ab4f89b3e6ec0df5e4f8d9551f3
raw CSV SHA-256: 48c3f67a3b126bca83d20d2167f895801ecb9d4ce25c24d4b68b11b70d356d9f
compressed CSV SHA-256: 37edfe3cbc7044c2ee2a2aa195584cdc43df7ebf8717ccd0cba9c14636457220
log SHA-256: 2a4bf52873fcb06abe2bb1641361271f4a28418eb08ef4b089e73861944a21da
```
