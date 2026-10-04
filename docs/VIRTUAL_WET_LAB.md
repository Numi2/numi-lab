# Virtual Wet Lab

`numi wet-lab` opens an experiment workspace inside NumiLab. NumiVivo owns
source preparation, native prediction and verification; NumiLab supplies the
local interface. The shared `WetLabExperimentAdapter` boundary supports paired-donor RNA response, spatial tissue transport and real spatial
gene perturbation. Each adapter owns its
units, model, observation contract, comparison and evidence status.

Choose a measured specimen, freeze a prediction, reveal held-out treated
measurements, inspect every baseline and gene, and replay the retained record.
Prediction receives the selected donor's untreated cells and training pairs
from other donors. The source snapshot contains treated observations for later
scoring, but they never enter the native model or query. This is a computational
holdout and UI reveal boundary, not a secure data escrow.

## Start

Prepare the pinned Kang assay using NumiVivo's
`Tools/VirtualWetLab/prepare_kang.py` (see that directory's README), then:

```sh
export NUMIVIVO_HDF5_LIBRARY=/absolute/path/libhdf5.dylib
numi wet-lab \
  --vivo-root /absolute/path/numiVivo \
  --binary /absolute/path/numivivo \
  --assay /absolute/path/kang/assay.json \
  --workspace /absolute/path/experiments
```

Open `http://127.0.0.1:8768`. `--port 0` selects an available port;
`--catalog` prints the list of assay catalogs without starting the server.
Repeat `--assay` to register molecular/spatial assays alongside RNA; the selector and
experiment history dispatch through each record's versioned adapter.
A spatial-only workspace does not require `--binary`.
`NUMIVIVO_ROOT`, `NUMIVIVO_BINARY` and `NUMI_PYTHON` can configure owner paths.
There is no queue, service account, network binding or second prediction engine.
The single-user server binds only loopback and validates host, origin and a
session token. Native calls run serially; retain failed records and inspect logs.

Each experiment gets a new directory. The registration is written before native
fitting; the prediction seal precedes comparison. The complete directory retains
source counts, preparation provenance, donor selection, native model, predictions,
owner hashes, logs and scored measurements. Downloaded JSON is a summary; keep
the directory and the exact executable/HDF5 runtime to replay. History reopens
existing experiments. A repeated deterministic run is not another biological
replicate, and changing donors starts a distinct experiment.

## Evidence and scope

The first adapter uses the public Kang 2018 benchmark: 2,651 source-labelled
B cells, eight paired donors, 15,706 measured genes, IFN-beta and the study's
six-hour observation. One donor is held out; seven donors train the existing
fixed-alpha context ridge. Native no-change, training-mean and training-median
predictions are retained. The primary comparison is all-gene RNA-response RMSE;
MAE and per-gene values are also shown. No model is chosen using held-out scores.

The October 3 acceptance ran all eight donors and independently reconstructed
raw-count aggregation, all predictions and RMSE with NumPy. Native replay and
tamper/unsupported-input checks passed. Context ridge improved RMSE over the
training mean by 0.074–4.065% and beat no change in each donor. This previously
inspected cohort supplies software and development-data evidence, not fresh
biological qualification. No new predictor was trained or promoted by this UI.
The exact acceptance receipt lives in NumiVivo's
`Tools/VirtualWetLab/evidence/2026-10-03/checks.json`.

Counts and labels are public measured data. Primary-library accession mapping
remains unresolved; batch is unreported. There is no arbitrary dose/time input,
calibrated probability, cell survival, growth, tissue function or clinical claim.
Source study: https://doi.org/10.1038/nbt.4042.

## Extending the workspace

Keep the select → predict → compare → record lifecycle. Each new assay family
must supply its own source admission, supported interventions/times and units,
control and replicate semantics, owner prediction command, sealed prediction,
independent observed endpoint, comparison and replay verifier. Do not reinterpret
RNA as phenotype or reuse RNA RMSE for unrelated measurements.

## Spatial tissue transport v1

Build `numi-matter-wet-lab` from the standalone `matter` CMake project and use
NumiVivo's `Tools/VirtualWetLab/prepare_spatial.py` to register an authored mesh,
material, transport/cell plan and native runtime identities. NumiVivo's
`Tools/VirtualWetLab/SPATIAL.md` contains complete build, registration, observation
and acceptance instructions. Add its prepared `assay.json` with another `--assay`.

The Matter exporter runs the existing native FEM runtime and publishes geometry
only after accepted steps. NumiVivo then constructs extracellular regions and
explicit cell compartments, diffuses a declared tracer across shared faces and
exchanges it through hypothetical passive membranes. Spatial pulse fields and
sample times are registered before execution; the matched control gets no pulses.
The current bridge transfers one frozen accepted geometry. Mechanics and transport
have separate clocks; this is not simultaneous or two-way biological coupling.

The compact UI exposes the registered specimen/protocol, control, prediction,
observation reveal, sampling-time selector, compartment field projection and
per-compartment values. Every arm/time enters scoring. Each record retains native
geometry, full fields, amounts, injected-amount ledger, hashes and transition
statuses; replay reconstructs both native owners. No-observation records keep
reveal unavailable. Measured and numerical-reference comparisons stay distinct;
neither automatically marks the tissue biology validated.

The shipped 12-cell/12-region example is an authored deformable scaffold using a
synthetic material, with two pulses and five sample times. Nine acceptance cases
passed, including ablations, independent matrix-exponential comparison, integrity
rejection and RNA legacy replay. Baseline maximum field error was 3.416e-6 mol/m3;
halving the time step reduced it to 7.954e-7. This is evidence about the conservative
discrete graph and software integration. Nonorthogonal continuum diffusion,
biological calibration, cell growth/function and mechanical feedback remain open.
The receipt is in NumiVivo's `Tools/VirtualWetLab/evidence/2026-10-03-spatial`.

Molecular target engagement still requires its own measured occupancy/kinetic
endpoint. The v0.2 gene-perturbation adapter below does not infer that mechanism. A future compound → molecular → cellular →
spatial chain must retain evidence for each transition instead of inheriting a
single validity flag. NumiBrain/NumiHuman context remains optional and belongs
only to assays that require it. A cancer workspace should reuse these owners.


## v0.2: an experiment-centred tissue workspace

Prepare NumiVivo's pinned GSE274447 assay with
`Tools/VirtualWetLab/prepare_spatial_perturb.py` and add its `assay.json` to the
same command. Full source, preparation, native checks and replay instructions are
in NumiVivo's `Tools/VirtualWetLab/V02.md`. No external UI frameworks or model
services are required. CMake installs the HTML, JavaScript and stylesheet together.

The main canvas uses measured cell centroids and source coordinates. Select a
cell to inspect its endpoint RNA, matched control reference, predicted regional
RNA, residual and nearby cells. Spatial proximity is shown without claiming a
biological interaction. Regional means plotted at cell locations are labelled as
regional expectations, never single-cell forecasts. A source geometry subset
provides context; only the admitted guide/control cells carry assay measurements.

Use Control / Predicted / Observed / Error to inspect the same readout. The gene
search, linked plots, residual map and failure-gene list stay synchronized.
“Select intervention” selects supported regions directly on the specimen; it
does not fabricate a new measured perturbation distribution. Unavailable regions
explain their admission failure. Pan and zoom preserve source coordinates.

The timeline scrubs only registered sample times. The real Clu experiment has
one destructive endpoint, so it does not animate a fictitious trajectory. The
synthetic transport adapter retains its five native times and compartment fields.
The old eight-donor assay remains available as an RNA regression.

“Compare with reality” verifies the seal before revealing held-out observations.
Comparison retains all-gene baseline RMSE, gene errors, region failures and
explicitly unavailable calibration. For the current held-out mouse the primary
criterion was **contradicted**: context ridge RMSE 1.877694 versus no-change
1.848305. One region with four controls and 18 target cells qualifies; regions
and deterministic repeats are not independent biological replicates.

Every view offers evidence/provenance inspection. The six evidence states are
MEASURED, SIMULATED — VALIDATED DOMAIN, SIMULATED — OUT OF DISTRIBUTION,
MODEL INFERENCE, HYPOTHESIS and UNAVAILABLE. Source, runtime hashes, transitions,
assumptions, holdout state, uncertainty and replay-directory identity are exposed.
No current v0.2 biological output is promoted to validated-domain status.

The command bar supports exact gene names, `gene Clu`, `knockout Clu`,
`region region-1-0`, `predict`, `compare` and `replay`. Unsupported targets,
concentrations and arbitrary durations fail explicitly. This is a bounded typed
compiler, not an unrestricted natural-language biology simulator. Templates save
supported selections; history retains immutable experiments. Compare experiments
shows two independent records and matched gene/region readouts without pooling
replicates or units.

Advanced Protocol & model exposes the typed transition plan and ExperimentCampaign.
A campaign preregisters a hypothesis, supported selection matrix, simulation
repeats and fixed success criterion. Every arm is sealed before any reveal;
failures remain records. This dataset supports one target, endpoint and region,
so dose/time sweeps remain unavailable. The next-region suggestion is model
disagreement, not calibrated information gain. Qualification exposes the Arc
2026 contract, whose official artifacts, scorer and zero-shot result remain open.

Browser acceptance covers the real prediction/reveal/replay loop, provenance,
selection, failure views, templates, campaign controls, legacy assay display and
narrow layouts. The native source/prediction oracle, campaign and legacy replay
receipts live in NumiVivo's `Tools/VirtualWetLab/evidence/2026-10-03-v02`.


## Intervention design development toward v0.4

The existing workspace accepts `--design-campaign` pointing to a sealed NumiVivo
multi-study campaign and `--receiving-campaign` pointing to explicit receiver
experiments. Define an RNA program, compare all supported interventions including
no intervention, seal the objective and candidates, reveal measurements, and
score the choice. Saved objective comparisons can be reopened in the same view.
Candidate plots and computed diagnostics share the experiment record.

The tissue canvas additionally inspects separate target/receiving-type response
distributions, with their own control-reference context, measured cells, model
errors and source evidence. Raw counts and normalized inference remain distinct.
This receiver corpus is permanently exposed development data. Unresolved section
identities prohibit a verified spatial-edge model. Dissociated studies have no
spatial preservation readout, so preservation objectives remain disabled.

The full requested v0.4 milestone is **not complete**. Biological promotion is
not earned: target-aware neural intervention choices did not beat the stronger
training ranking on the newly reserved cross-assay cohort. The frozen v0.3 State
comparison remains PerturbMean, not State Transition. See NumiVivo's
`Tools/VirtualWetLab/V04.md` and `evidence/v04` for executed campaigns, all ablations,
source metadata, sparse-program detection, underfitting diagnostics, replay and
remaining scientific gates. The UI does not supply biological models.

Local Chromium qualification on the actual Apple Mac16,12 (24 GB) measured
2.532 seconds to a ready workspace in a fresh server process and 7.055 seconds
to import 123,921 measured cells. The filesystem cache was not cleared. Candidate
preview/seal/replay requests took 45/152/729 ms; observation reveal took 5.584 s.
Cached picking/drawing timing is retained separately, not substituted for loading.
The 390-pixel viewport has no horizontal document overflow. Observation-snapshot
interruption recovery, native replay, and v0.2/v0.3 regression pass.

Safari qualification is **blocked**, not passed: its remote-automation setting is
disabled and native UI access failed. Hardware/browser generalization and the full
software release gate remain unqualified. Evidence is under
[`docs/evidence/virtual-wet-lab-v04`](evidence/virtual-wet-lab-v04/README.md).

The [verified development package](https://github.com/Numi2/numiVivo/releases/tag/virtual-wet-lab-v0.4-development) contains nine replayed native candidates. The installed launcher now includes intervention design and receiver inspection while retaining the RNA, Clu, learned-spatial and transport assays and their existing records.

## Shared Codex workflow (v0.4 recovery)

The installed NumiLab plugin has a dedicated
[Wet Lab workflow](../plugins/numi-lab/skills/numi-lab/references/virtual-wet-lab.md).
Select a population, then use `numi wet-lab context` and a typed `propose` command.
Cards, human edits, persistent candidate comparisons and Codex actions share the
same revision-checked state beside the tissue canvas. Read context before each
write; stale writes fail rather than overwrite a human edit. Seal and execute
asynchronously, cancel/recover stopped operations, explicitly authorize reveal,
and follow residuals to the relevant gene and population. Sealed experiments are
immutable; use `revise`, not undo, to change their intent.

See [executed qualification](evidence/virtual-wet-lab-v04-recovery/README.md).
Training-fit qualification, reserved cellular results and spatial validation are
separate. The original spatial model is retained as an experimental regression.
