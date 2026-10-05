# Accepted breath event observer

The lung-compliance x0.8 sensitivity trace exposed two positive-flow excursions
entirely between 32 ms retained samples. GPU event capture showed that these
were round-off sign changes near zero (below 0.00001 ml/s), which the old raw
sign counter mistakenly counted as breaths. The original failed analysis and
the raw event-ledger diagnostic are retained on the SSH Mac mini under
`respiratory-reference-sensitivity-001` and
`breath-ledger-sampling-regression-001`.

The observer now uses a 0.001 ml/s Schmitt threshold and retains accepted event
step/time and a compensated inspired-volume ledger. This threshold applies
only to the breath observer. Airflow is not clipped, and mechanics, gas
transport and Brain regulation do not use these observer fields. Complete
breath metrics use the GPU event ledger even when presentation samples do not
resolve a crossing. Legacy traces retain the strict interpolation fallback.

On the Mini, the 12-second x0.8 compliance regression resolved one complete
5.134-second interval: 11.6868 breaths/min and 5.92271 L/min. The raw sign
counter had reported 34.1168/min. All 30 retained non-observer columns are
identical, including volumes, flow, pressures, blood/gas accounting and gas
observations. The 128-step full native rejection/replay probe also passes;
all 42 pre-existing trace columns, surface CSV and terminal body state match
the pre-change control exactly. Twelve adapter tests pass.

This is an observer correction and numerical regression, not anatomical,
five-minute whole-body or clinical qualification. The candidate native build
still uses the earlier diagnostic anatomy. Runtime hashes and invocations are
retained here; full traces, logs and recordings remain under
`/Users/n/numi-human-resting-evidence-20261005/` in
`breath-ledger-hysteresis-regression-001` and
`native-breath-ledger-hysteresis-transaction-001`. All execution used SSH Mac
mini (Apple M4 Pro); the Air was used only for source edits and transport.
