# Native scientific workflow example

This example builds a measurement instrument, makes a quantitative prediction,
runs paired native experiments, revises the model and tests the new prediction
on unused seeds. It uses Numi's owning synthetic-culture implementation on Metal.
It does not implement a substitute simulator or establish biological learning.

Read [the workflow and command contract](../../docs/SCIENTIFIC_WORKFLOW.md).
`build_neuron_instrument.py` compiles the native owner sources into an isolated
binary and retains its build receipt. `neuron_weight_instrument.py` validates and
extracts the native mean plastic weight, retains raw owner output and verifies
that STDP-off preserves the authored 0.05 control. `prepare_neuron_study.py`
calibrates the reader and authors the prediction and paired trials.

## Retained execution, 2026-10-03 Europe/Oslo

The complete compact records are in [evidence/20261003](evidence/20261003).
Local timestamps in the records are UTC on October 2. The focused owner build
ran on Apple M4 and passed its native parity, replay and transaction checks;
see [the native qualification](evidence/20261003/native-qualification.json) and
[source/compiler/artifact receipt](evidence/20261003/native-build-receipt.json).

| Study | Preregistered mean treatment minus control | Observed mean | Result |
| --- | --- | --- | --- |
| Discovery, seeds 2201–2203 | [0.0001, 0.01] | 0.00007634277 | Contradicted |
| Follow-up, untouched seeds 2301–2303 | [-0.0001721588, 0.0003248443] | 0.00015689637 | Supported at this scope |

Units are native authored synaptic-weight units. All twelve trials completed;
the native quick protocol produced 12 windows per trial. Every pair matched
seed, topology, starting-state fingerprint, current scales and window count.
The independent experimental unit is the seeded culture, not a tick or neuron.

The first approximation predicted a consistently positive net potentiation
large enough to exceed 0.0001 on average. One discovery pair decreased, and the
mean fell below that minimum. The [revision](evidence/20261003/discovery/study/revision.json)
replaced that lower bound with an empirical small-effect model and seed variation.
The next interval was fixed to discovery mean ± three discovery sample SDs
before the follow-up trials. This is an explicitly heuristic predictive range,
not a confidence interval, calibrated coverage claim, or statistical discovery.
The held-out result retained the parameters and recorded their limited validation.

The record preserves the [initial plan](evidence/20261003/discovery/plan.json),
[initial analysis](evidence/20261003/discovery/study/analysis.json),
[follow-up plan](evidence/20261003/confirmation/plan.json),
[follow-up analysis](evidence/20261003/confirmation/study/analysis.json), and each
trial's exact raw native JSON, extracted measurement, logs, commands and hashes.

## Instrument failures are also evidence

The older installed binary did not emit `mean_plastic_weight`. Six early trials
therefore failed extraction and produced no valid scientific observations.
Their ignored local directory was subsequently removed by sparse-checkout
expansion; those early raw files are **not retained evidence**. The old source
checkout also failed its strict shader build on an unused parameter; current
`main` already contained the fix. No physics changes were needed for this study.

The separately preregistered [recovery check](evidence/20261003/recovery-check/plan.json)
is a known-failure diagnostic, not a reconstruction or novel scientific result.
Its retained native output reproduces the missing field. The notebook records
the invalid trial, a reasoned stop, the missing second arm, and an inconclusive
analysis. All new durable evidence was stored outside the checkout before this
compact archive was made.

## Verify without the native runtime

```sh
numi science verify examples/science/evidence/20261003/discovery/study
numi science verify examples/science/evidence/20261003/confirmation/study
numi science verify examples/science/evidence/20261003/recovery-check/study
```

Verification checks recorded integrity and recomputes analysis; it does not
rerun Metal or prove biological validity. Raw native observations live under
each study's `trials/*/output/`. The exact build executable remains at the local
path in the build receipt; reconstruct it from the recorded owning sources to
run on another Mac. The archive retains absolute source/runtime paths as provenance.

This example establishes an executed scientific workflow and a bounded weight
effect in a small unconditioned synthetic protocol. Full-size learning,
long-horizon adaptation, biological calibration, and population-level inference
remain separate studies.
