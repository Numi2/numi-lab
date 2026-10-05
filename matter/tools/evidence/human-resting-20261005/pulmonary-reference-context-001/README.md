# Supine pulmonary pressure reference context

The unchanged 320-second native endurance trace had mean pulmonary artery
pressure 14.759923 mmHg over [290, 320) seconds. It remains below the previously
selected AACN 15–20 mmHg comparison interval. The observation parser now also
reports the posture-specific healthy-subject context from [Kovacs et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19324955/),
DOI 10.1183/09031936.00145608: supine mean 14.0, SD 3.3 mmHg. These are published
aggregate descriptive values, not individual normal bounds or a new pass gate.
Neither the physical model nor any previous comparison was changed.

All 16 existing adapter regression tests passed on the SSH Mac mini with Python
3.13. The first test attempt had the wrong package search path and failed at
import; its log is retained. The corrected command used
`PYTHONPATH=/Users/n/numi-human-resting-integration-20261005/numi-lab/python` and
the isolated analysis source directory. `retained-endurance-context.json`
binds the immutable trace and exact analysis source by SHA-256.

This evidence is a literature comparison and parser regression, not a new
simulation, anatomical acceptance, physiological calibration, or clinical
validation.
