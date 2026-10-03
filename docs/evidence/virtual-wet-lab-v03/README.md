# Virtual Wet Lab v0.3 qualification log

First working increment: the existing specimen renderer now caches cell and gene
lookup maps and selected sparse gene buffers. Picking uses a spatial grid.
Control, predicted and observed layers share their comparison scale; residuals
use a symmetric legend explicitly labelled `predicted − observed`.
Raw control UMIs remain distinct from normalized regional predictions.

Verified locally on the retained v0.2 specimen (2,514 displayed cells): browser
load, canvas construction, cached lookup and selection index. Node checks cover
sparse zero versus missing values, buffer reuse, nearest-point picking and
shared/signed scales. This is not the larger-specimen performance qualification.
The old experimental records and biological verdicts are unchanged.

## Qualified learned-response workspace

The native NumiVivo assay is integrated into the existing laboratory. A researcher
selects a control-reference population, compares multiple interventions, switches
between direct and neighboring responses, seals every selected arm together,
reveals measured RNA, inspects computed failures and replays the native tensors.

Four canvases share cameras, selection, gene and comparison scales. Population
means/distributions are drawn as population envelopes rather than unique cell
forecasts. Disconnected spatial groups are not joined across the empty chip.
Observed cell RNA and raw UMIs remain distinct from normalized inferred means.
The cell inspector exposes guide assignment, uncertain projected identity,
reference coverage, and a measured selected-cell value only when it is available
within the selected comparison. Negative barcode is not assumed untreated.

Linked marginal distributions show the common control, native predictive mean
and uncalibrated spread, and unique measured outcome cells after reveal. A gene's
campaign effect ranking links back to population selection. Persistent comparisons
can be reopened with their exact arms and record. Unsupported targets retain their
exclusion reasons. Completed-campaign cell-type diagnostics are explicitly separate
from the new comparison's reveal. No suggested biological mechanism is invented.

`browser-workflow.json` records a six-arm Cfap410/Fasn/Gfap experiment, its hidden
observations before reveal, common reference, signed residuals, neighbor selection,
computed failure diagnostics and exact native replay. Desktop and 390 px mobile
views were inspected; mobile document width is 390 px without horizontal overflow.
The v0.2 workspace remains selectable, with the legacy source-file/animal-identity
ambiguity corrected in the explanatory UI while preserving the original record.

`browser-performance.json` measures the actual 123,921-cell training specimen on
this Apple-silicon Mac, with all source cells indexed, a full-specimen LOD raster
and four canvases. Headless Chromium 154, 1280 × 800 viewport:

| Interaction | p95 | Predeclared target |
|---|---:|---:|
| Warm gene switch, 40 samples | 0.8 ms | ≤100 ms |
| Indexed picking, 500 samples | below browser timer resolution | ≤20 ms |
| Pan frame interval, 89 samples | 16.7 ms | ≤33.4 ms |
| Four-canvas drawing script | 0.5 ms | diagnostic only |

These are cached local-browser measurements, not cold network-load, Safari,
all-hardware or biological qualification claims. The test script is retained in
`python/metalrobo/wet_lab/qualification/`.

**Software qualification passes; biological promotion fails.** The learned model
has held-out equal-target RMSE 0.761057 versus no-change 0.736076; removing or
shuffling neighborhoods changes it only slightly. Four isolated test controls
fall below the required five. Unseen-training-target exploration is not clean
unseen-perturbation qualification because that target entered validation selection.
The full source-bound outcomes, weights and qualification audit belong to NumiVivo:
https://github.com/Numi2/numiVivo/blob/main/Tools/VirtualWetLab/V03.md

The official Arc six-metric evaluator now has an executed, separately recorded
local CRISPRi path. This is not a returned VCC2026 challenge score or spatial model
validation. No paid CI infrastructure was added.
