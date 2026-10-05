# Displayed accepted-state trace admission

The existing native intervention adapter now checks every displayed-state row,
including the initial frame and final partial segment. It requires the native
cadence (0, 31, 63, …, N−1), time `step * dt`, finite positive functional volume
targets, no invalid skin vertices, no skin–bed penetration beyond the existing
1 mm inspection threshold, and rendered volume relative error at most the
existing GPU threshold of 2e-4. This is numerical geometry consistency; tissue
interface qualification remains false.

The renderer publishes the captured pre-step state only after the next physical
step succeeds. Its own same-frame target columns are used. Comparing against
the next coupled CSV row is invalid.

Validation ran only on `ssh macmini`, using
`/Users/n/numi-human-prep-venv-20261005/bin/python`, in the candidate source
checkout. `python -m unittest test_resting_intervention_study -q` passed all
16 tests, including missing/intermediate-invalid frames, wrong clocks, volume
error, and a final partial segment. No simulation was rerun for this parser
change.

The retained `cardiac-native-admission-002/resting-surface-audit.csv` passed:
95 displayed accepted frames; minimum full-skin bed gap −0.000110551714897 m;
maximum functional volume relative error 1.16924240956e-6. Exact input identity
is retained in that native run and the accepted-render geometry evidence.

The older `native-constraint-projection-30s-nextafter-001` trace lacks the later
same-frame diaphragm/rib/volume target columns. It is explicitly ineligible for
this new admission check; no missing fields were fabricated or inferred from
another clock.
