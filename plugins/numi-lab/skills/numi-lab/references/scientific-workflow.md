# Standard scientific workflow

Numi Lab is an instrument-building laboratory. Codex acts as the researcher:
it asks questions, authors instruments, proposes models, designs experiments,
interprets evidence, and revises those models. Native suite owners retain
execution and physical authority. This is the default for scientific work in
every Numi domain; it is not a fixed experiment catalog or a second simulator.

## 1. Frame the question and current model

State the observable question, competing explanation, existing model/version,
assumptions, and what evidence could change the model. Identify the owning
runtime from live help. Inspect existing studies and active processes before
launching. Separate old/exploratory evidence from a new confirmatory experiment.
Select the next experiment for its ability to distinguish the explanations.

## 2. Build and qualify the measurement instrument

Reuse a trustworthy native observable if it exists. When it does not, implement
the missing probe, sensor, exporter, or analysis tool in its lowest owning
layer. Preserve causal timing, accepted-state publication, units, and physical
ownership. The instrument observes the experiment; it must not insert the
expected result or alter the intervention covertly.

Bind source, executable, model, parameters, inputs, dynamic libraries, shader
libraries, calibration artifacts, and environment settings by identity. Check
known-value references, null/positive controls, resolution, numerical stability,
and failure detection as applicable. Validate measurement semantics separately
from the scientific prediction. A parser calibration does not establish sensor
accuracy or biological validity. If a measurement is unavailable, build it or
report that limitation; never replace it with a reward, fixture, or animation.

## 3. Preregister a falsifiable prediction

Before collecting the test observations, retain an immutable plan containing:

- Question, model/version, mechanism and alternative explanation.
- Named observable, units, quantitative effect or interval, and refutation rule.
- Intervention, matched control/ablation, held-fixed variables, independent
  experimental unit, seeds/conditions, order/randomization, and repeats.
- Calibration and validity checks; missing, rejected, and failed-trial policy.
- Analysis/uncertainty method, sample budget, stopping rule, and unused follow-up
  conditions. State exploratory versus confirmatory status and evidence limits.
- Exact owner commands and relevant source/runtime/input identities.

Do not count frames/ticks from one trajectory as independent replicates. Use
paired seeds and identical initial state where the owner permits. Blind labels
or randomize order when drift or analyst choice could affect the outcome.
Controls must test the mechanism, not simply provide another successful run.

For a bounded scalar paired study, inspect `numi science --help` and the runtime's
`docs/SCIENTIFIC_WORKFLOW.md`. Author the plan using the documented JSON format,
then `numi science register PLAN STUDY`. This seals the prediction and artifacts
before `numi science run STUDY` executes the next declared trial. The notebook
is a recorder over owner executables, with no scheduler, hosted queue, or model
that decides what the researcher should do.

For designs beyond its mean-paired-difference analysis, implement and calibrate
the appropriate owner analysis and preregister its method. Do not flatten a
dose response, causal time series, uncertainty model, or multi-factor design to
fit an inadequate summary. Retain an equivalent plan/raw-data/revision record
if the notebook capability is unavailable; never silently skip the loop.

## 4. Run the controlled experiment

Use isolated outputs and the exact registered inputs. Verify runtime/device,
active workload ownership, resource budget, and the declared safety boundary.
Invoke the native owner; do not add a competing per-step host simulation.
Run every declared arm and retain logs, accepted/rejected counts, failures,
timeouts, seeds, environment, and raw measurements. Do not retry an unfinished
trial merely because observation timed out; inspect its live handle first.

Do not silently change thresholds, exclude a failed arm, increase the sample
budget until a claim passes, or treat uncalibrated output as a valid observation.
If an instrument or model changes, preserve the old study and preregister the
new version. Hardware still requires its owning arming policy.

## 5. Compare evidence with the prediction

`numi science analyze STUDY` uses every declared trial and compares the paired
effect with the sealed interval. Invalid evidence yields **inconclusive**, not
support. A valid out-of-interval result is **contradicted** and remains useful.
`numi science verify STUDY` checks retained hashes and recomputes the analysis.

Inspect the actual raw data and control validity, not only the verdict. Report
effect size, units, individual pairs, variation, failures, and confounders. The
built-in observed range and sample SD are descriptive; they are not population
confidence intervals or proof of a general mechanism. Distinguish instrument
failure, model disagreement, insufficient power, and an unidentifiable design.

## 6. Revise the model and test again

Record the prior model, cited analysis hashes, retain/revise/reject/inconclusive
decision, exact parameter/structure/assumption change, rationale, remaining
limits, and the next distinguishing prediction. If the model is executable,
change and hash its owning implementation/parameters as well as its statement.
Do not rename a failed model and preserve its unsupported behavior.

`numi science revise STUDY REVISION` retains the previous model. Register the
follow-up with `--parent STUDY`; it must use that revised model. Freeze the next
prediction before running unused seeds, conditions, or held-out data. Do not
call fit data independent validation. Continue until the requested scientific
question is resolved at its stated scope, or retain the precise unmet condition.

A completed report leads with what was learned, what changed in the model,
and whether the new prediction survived its independent test. Link the plan,
raw observations, analysis, revision and follow-up. Software, numerical,
simulation, physical, biological, and performance evidence remain distinct.

## Native worked example

The runtime's `examples/science` contains a calibrated reader for synthetic
neuron weight measurements and a study authoring script. It compares the native
quick protocol with STDP enabled against a matched STDP-off control. It is a
small simulation study from unconditioned weights, not learning qualification
or a reproduction of biological experiments. Read its current evidence before
choosing unused follow-up seeds; do not rerun it to manufacture novelty.
