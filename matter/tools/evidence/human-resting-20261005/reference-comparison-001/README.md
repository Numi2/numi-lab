# Descriptive physiology comparisons

The existing intervention observer now reports source-linked physiological
comparisons beside numerical conservation and acceptance. No reference value
is written into the simulated state, and an out-of-range observation is
retained rather than causing numerical rejection or being relabeled normal.

Ranges were checked on 2026-10-05 against AACN *Normal Ranges*, pages 9–12
(adult at sea level), and MedlinePlus *Vital signs* (reviewed 2025-01-01).
Supine male cohort context comes from Table 2 of Mendes et al.,
DOI 10.1016/j.bjpt.2019.02.007. Cohort mean/SD is explicitly not a normal
interval; OEP chest-wall volume is also not identical to airway gas volume.
Direct source links accompany every comparison in the generated observation.

The retained 30-second native diagnostic provides an executable regression
fixture. Its 10–30 s interval is below the general breathing-rate reference,
slightly above the PaO2 reference, and slightly below the pulmonary mean
pressure reference. Those discrepancies remain visible. Output, stroke volume,
arterial CO2, saturation, mean aortic pressure and heartbeat rate are inside the
selected intervals. This is neither a steady-state nor a five-minute
qualification, and none of these comparisons establishes clinical validity.

Fourteen existing/extended adapter tests passed on SSH Mac mini. New tests
check that outliers remain reported and that unavailable complete-cycle rates
are not converted into zero. The fixture trace is identified by exact path and
SHA-256 in the retained JSON. All computation ran on the Mini.
