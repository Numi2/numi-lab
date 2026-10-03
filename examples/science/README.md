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
qualifies the exact native binary and reader before authoring paired trials.
`weight_effect_model.py` generates predictions from JSON parameters, fits revisions
to verified observations, and records retained parameters after an independent test.
New studies use v2; the v1 records below remain historical evidence.

## Workflow integrity audit and fresh v2 execution

The v2 runner closes five reproduced failures: nonexistent calibration accepted,
lost trial directories silently rerunnable, reused training seeds accepted as
confirmation, parent evidence left unverified, and symlink output preventing a
terminal receipt. It also binds executable model predictions, snapshots inputs
and parents, handles interrupted process groups, and preserves the existing v1
runner for already registered work. The software regression suite exercises
these failures, archive relocation, executable fitting and v1 continuation.

The [audit and measured results](evidence/20261003-v2-summary.json) retain the
baseline reproductions and exact new record hashes. The complete portable v2
notebook is [20261003-v2.tar.gz](evidence/20261003-v2.tar.gz), including native raw
observations, calibration, input snapshots, model files and both parent studies.
The summary binds the archive SHA-256. Extract into a new directory and verify:

```sh
tar -xzf examples/science/evidence/20261003-v2.tar.gz -C /path/to/new/archive
numi science verify /path/to/new/archive/final-confirmation
```

Evidence archives are in the source repository; installed example packages omit
them. Their [published source location](https://github.com/Numi2/numi-lab/tree/main/examples/science/evidence)
is the same as this directory.

| V2 study | Frozen predicted mean interval | Observed mean | Result |
| --- | --- | --- | --- |
| Discovery, seeds 3201–3203 | [0.0001, 0.01] | -0.00001229190 | Contradicted |
| First confirmation, seeds 3301–3303 | [-0.0001512331, 0.0001266493] | 0.00015583143 | Contradicted |
| Final confirmation, seeds 3401–3403 | [-0.0002069455, 0.0005186084] | 0.00008823733 | Supported at this scope |

All 18 registered native trials completed on Apple M4 with matched controls and
12 windows. Separate calibration seeds 3200, 3300 and 3400 passed the frozen
0.05-weight control; native parity, replay and transaction qualification passed
before each study. No test seed was reused across the three batches.

Each revision executed the same explicit fit rule: mean ± three sample SDs of
the immediately preceding valid batch. Prior exposed units remained in the model
history even though the fit used the current batch. Each new prediction was
executed and sealed before unused observations. The first fitted interval failed,
revealing that three discovery seeds understated later variation. A second
revision survived one final batch; its parameters were retained unchanged.
Both contradictions remain in the archive. This small adaptive sequence is a
workflow demonstration and descriptive synthetic result, not calibrated
population coverage, a significance claim, or proof of biological learning.

## Historical v1 execution, 2026-10-03 Europe/Oslo

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
