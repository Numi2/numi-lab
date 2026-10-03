# Virtual Wet Lab

`numi wet-lab` opens an experiment workspace inside NumiLab. NumiVivo owns
source preparation, native prediction and verification; NumiLab supplies the
local interface. The first adapter is a paired-donor RNA-response assay.

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
`--catalog` prints supported specimens without starting the server.
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

Molecular target engagement should route to NumiVivo's engagement owner and
compare a measured occupancy/kinetic endpoint. Spatial tissue experiments need
the cell/tissue owner plus supported transport/mechanical coupling and a measured
spatial endpoint. Neither family is runnable through this first workspace.
NumiBrain/NumiHuman context is optional and belongs only to assays that require
it. A cancer workspace should reuse these owners and records.
