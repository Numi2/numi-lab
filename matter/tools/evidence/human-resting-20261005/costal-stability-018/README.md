# Integrated native stability diagnostic, 2026-10-05

The native whole-body scene completed 160,000 accepted 2 ms steps on the SSH
Mac mini (Apple M4 Pro): 320.000015 simulated seconds, including 310 seconds
after the declared 10-second initialization window. It recorded 59 breaths
and 373 complete filling/ejection cycles with matched physical/physiological
clocks and no root assistance. No inputs changed during execution.

Native wall time was 3,200.714700 s, or 0.099978 times real time. Independent
CPU asset preparation was active; this is not an isolated performance
benchmark. The unretimed recording has all 5,001 displayed accepted frames,
last presentation time 3,200.913333 s, and duration 3,201.575 s. Its exact
Mac-mini path and SHA-256 are in `retained-artifacts.json`.

Maximum blood-volume error was 0.099652 mL of 5,150 mL. Maximum reported O2
and CO2 amount-balance residuals were 0.000233 and 0.000466 mL STPD.
Rendered functional-volume relative error stayed below 2.052e-6; ventricular
material-volume relative error stayed below 1.952e-6. Material and functional
geometry guards remained clear. These guards do not test all intersections.

The last 30-second window measured mean PaCO2 39.847 mmHg, PaO2 99.450 mmHg,
SaO2 97.182%, aortic pressure 89.101 mmHg, pulmonary pressure 14.760 mmHg,
and forward aortic output 4.821 L/min. Complete-breath measurements gave
11.512 breaths/min and 6.149 L/min inspired ventilation. The unchanged
reference comparisons retain the respiratory-rate failure against 12–18/min
and pulmonary-pressure failure against 15–20 mmHg, alongside the separately
identified supine cohort/review context. These are plausibility comparisons,
not anatomical or clinical validation.

## Explicit failed and unresolved gates

This scene is **not anatomically accepted**. Independent exact checks of the
same input scene's native017 accepted frames found 12 crossing lung–bone
pairs (2,918 triangle pairs) at step 639 and nine crossing pairs (966 triangle
pairs) at step 2783. They involve actual ribs/sternum, not intended pleural
interfaces. Both complete 170-pair audit results are retained here. An earlier
working summary incorrectly reported source-fit clearance: re-reading the
original fit013 report shows the same totals (2,918 and 966). This was a
reporting error, not a demonstrated source-to-native clearance regression.
`source-fit-013-report-correction.json` binds the original report and its
per-pair counts. Cardiac-wall and liver interfaces also
remain under repair. Zero runtime guard status does not waive these defects.

The body's center of mass moved after initialization, including about
18 mm laterally during seconds 140–170. The final 30 seconds showed less
than 0.49 mm range on each axis. The trace records this motion; stationary
rest and absence of unexplained drift have not been established.

This was an engineering stability run, not the final preregistered paired
respiratory-intervention study. That study and a corrected complete-cycle
anatomical audit remain required. No final delivery claim is made.

## Reproduction and evidence

`invocation.json` contains the exact launch arguments, environment, device
owner fingerprints, binary/library/shader/configuration/asset hashes,
source revision and source-file hashes. `source.diff` binds the frozen
source beyond its base revision. The original accepted CSV files are
stored losslessly compressed here, with uncompressed hashes in the analysis.
`analyze-costal-fit-stability-018.py` runs on the Mac mini and reuses the
existing native study collector; it does not implement simulation physics.
