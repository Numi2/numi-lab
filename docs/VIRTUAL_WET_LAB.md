# Virtual Wet Lab

`numi wet-lab` opens an experiment workspace inside NumiLab. NumiVivo owns
source preparation, native prediction and verification; NumiLab supplies the
local interface. The shared `WetLabExperimentAdapter` boundary supports both
paired-donor RNA response and spatial tissue transport. Each adapter owns its
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
Repeat `--assay` to register a spatial assay alongside RNA; the selector and
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

Molecular target engagement should become a separate adapter that compares a
measured occupancy/kinetic endpoint. A future compound → molecular → cellular →
spatial chain must retain evidence for each transition instead of inheriting a
single validity flag. NumiBrain/NumiHuman context remains optional and belongs
only to assays that require it. A cancer workspace should reuse these owners.
