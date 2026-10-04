# Virtual Wet Lab: shared cellular and tissue experiments

Use the existing `numi wet-lab` workspace and NumiVivo owners. This is a local,
shared experiment surface, not a separate assistant, simulator, or hosted queue.

1. Read `numi wet-lab context`. The active workspace is discovered automatically;
   do not ask for assay paths or JSON. If it is unavailable, start `numi wet-lab`
   in a persistent local terminal, then open its printed local URL. The user
   selects a population in the admitted cellular dataset or tissue. Context returns explicit assay, specimen,
   population, experiment identifiers and a workspace revision.
2. Turn the user's molecular goal into the exact gene list they requested. Use
   `numi wet-lab propose --revision REV --genes GENE ... --targets TARGET ...
   --title 'Exact molecular objective'`. Supported targets and measurement
   coverage appear on the editable card in the same biological workspace. An absent
   marker is UNAVAILABLE, not zero. A predicted quantity is MODEL INFERENCE.
   Never silently replace an unsupported objective with a subset. Register a new
   objective explicitly if the user requests that change.
3. Read context again before **every** edit or execution. Human edits are
   authoritative; use the returned revision with `edit`, `undo`, `seal`,
   `cancel`, `recover`, `revise`, `replay`, and `reveal`. A stale-write rejection
   requires reading and discussing the changed intent, not automatically
   resubmitting the old edit. `edit --id ID --genes ... --targets ...` updates the
   visible card. Dragging candidates into the comparison updates this same state.
4. Review exact objective coverage and control counts. Unsupported spatial
   preservation stays disabled until sections and receiving controls are verified.
   Current spatial specimens are exposed development data; do not call this
   workflow a newly independent validation. A reliable winner is unavailable
   without qualified transfer uncertainty.
5. `numi wet-lab seal --revision REV --id ID` starts the existing native owner
   asynchronously. Inspect progress with `context`; `cancel` preserves partial
   artifacts. `recover` is allowed only after the original process has exited.
   Sealed experiments are immutable. Use `revise` to create an editable descendant.
6. **Never reveal observations while inspecting candidates.** Only after the
   user explicitly authorizes opening measurements, run `numi wet-lab reveal
   --revision REV --id ID --authorized-reveal`. A request to prepare, inspect,
   compare or seal is not reveal authorization. The human may instead click
   “Authorize reveal of measurements” on the card. This explicit operation opens
   the existing owner's observation path.
7. Click a predicted effect or a failed-objective diagnostic to locate its
   population and molecular evidence. Explain computed residuals and baseline
   comparisons, not an invented mechanism. Use `replay` and `history` to inspect
   retained evidence. Report training-fit qualification separately from reserved
   experimental results and biological promotion. State PerturbMean is not the
   learned State Transition comparator.

Draft edits support undo. Operation progress and experiment history survive
browser reload. A fresh Codex session reads the same state without file paths.

## Compare exact model versions and supported conditions

`context` now lists registered models, compatible specimens/populations and
supported conditions. Select explicit identifiers from that response; never
substitute v0.3 when a corrected model cannot run. The v0.3 assay remains a named
regression. Cards bind the assay, runtime, weights, specimen and population.

Author the objective once with `propose --models MODEL_ID ... --conditions
CONDITION_ID ... --genes GENE ... --targets TARGET ... --revision REV`.
Use `edit --id ID --models ... --conditions ... --revision REV` to change the same
comparison. Model and condition edits participate in undo and stale-write checks.
All selected model/condition arms are sealed before a separate authorized reveal.
Read each axis's `canExecute`, `reason`, `corrections`, `coverage` and `binding`.
A blocked condition is not a zero response. Only apply a supported correction
consistent with the user's intent; changing objective genes is an explicit edit.

The current spatial source supports its measured endpoint only. It has no
qualified tunicamycin/thapsigargin/DMSO tissue conditions or spatial-preservation
objective. Those cellular conditions must not be relabeled as tissue inputs.
The corrected spatial model is exposed development, not biological promotion.
Model switching retains the candidate, gene, camera and shared comparison scale.
`axisResults` holds the exact record and objective evaluation for each selected
model/condition. Report numerical ranking changes separately from evidence of a
reliable winner. Inspect gene residuals and no-change/matched baselines after an
authorized reveal; replay every axis using its retained native runtime.

## Decision support in the specimen workspace

The default is one large specimen view. Use “Compare synchronized views” when
control, prediction, observation and signed residuals need to be inspected together.
The canvas supports pointer pan/pick, arrow-key pan, plus/minus zoom, zero reset
and Enter to inspect its center. Touch drag/pinch handlers exist; physical touch
qualification is still pending, as is an uncoached researcher session. Neither
browser automation nor model agreement is a usability/confidence qualification.

`context` includes `selectionSupport` and each card's `decisionSupport`. Inspect
selected-population and candidate-specific reference/outcome counts, barcode-negative
exposure uncertainty, source-bound units, target overlap and projected population
support before describing any numerical recommendation. Specimen totals are not
local controls. The observed Gfap arm in the retained neuron demonstration has one
cell, no qualified recommendation, and no fitted training-ranking comparator.

Coverage updates after draft edits. Unknown genes and interventions remain in the
request with blocked execution and an explicit correction. Never silently remove
them. Correction buttons create ordinary revision-checked edits, covered by undo.
Model differences on sealed cards decompose objective utility into gene contributions
and retain the same specimen, candidate, gene and scales. These are numerical
attributions, not mechanisms. Clicking “Inspect failed objective” exposes measured
values and model/no-change/matched-baseline residuals without rebuilding the comparison.

The experimental v0.4 baseline is frozen. The subsequent ESM2 experiment separates
held-target and held-context technical development folds; it does not establish
spatial or independent biological validation, and did not replace the registered
spatial model. Do not describe 69/81 training groups as unseen genes, or one stimulated
Jurkat validation capture as validation across all studies. Keep researcher usability
explicitly pending per the user's instruction.


## Cellular experiments through the same shared cards

Discover the registered assay's `presentation`, populations, conditions and readouts
in `context`. Cellular assays use a population view; tissue assays keep their spatial
canvas. Geometry is optional. Do not invent cell positions or cell-level distributions
for a model that only provides population moments. A frozen cellular artifact is
explicitly identified on the card; there is no substitution with a spatial model.

Select the dataset, population, supported condition and gene in the shared workspace,
then use the same `propose → edit → seal → authorized reveal → replay` operations.
A condition name on a card is not support: read the owner qualification before sealing.
The exact objective, feature axis, reference population, condition and model must be
registered before native inference starts. Unknown conditions remain blocked requests.

“Explore accessible measurements” requests only observations the owner already permits.
It does not authorize a reveal. “Predict a response” previews or executes the registered
model, with its observation boundary unchanged. Gene requests are bounded and the
browser does not receive the full expression matrix. Changing population, model,
condition or access state invalidates the corresponding view; obsolete responses must
not overwrite a newer selection. The interface displays owner-computed utility and
residuals rather than implementing its own scientific score.

Read a fresh `context` before each agent action. A rejected stale edit is not accepted
optimistically: retain the current human edit, inspect it, and revise only in accordance
with the user's intent. Cards are editable before execution; sealed experiments remain
immutable, with changes expressed as new revisions. During execution, inspect progress,
cancel if requested, and recover only after the original operation has stopped. A
restored operation must be inspected as interrupted rather than resumed automatically.

A cellular end-to-end software demonstration is not a claim of transferable biology.
The first registered cellular model reuses retained native weights and does not establish
a benefit from population input. Multi-cell versus mean-input learning, heterogeneous
source ingestion and independent evaluation each require their own executed evidence.
Safari, physical touch and uncoached researcher usability remain separately qualified;
the researcher usability session is explicitly pending.


## Portable cellular experiments and restoration

Use `numi wet-lab catalog` (or `catalog --assay ID`) to discover admitted assays,
features and supported populations without supplying paths. For a cellular assay,
select the exact source/context with `select --assay ID --specimen ID
--population ID --condition ID --revision REV`. Condition is an execution input;
an arbitrary condition label is rejected. Use the same proposal/edit/seal/reveal/replay
operations as tissue experiments. Objectives are preregistered before native inference.
The first frozen cellular artifact exposes population means from already-exposed
technical development data; individual-cell distributions are unavailable.

Use the existing CLI's named operations, discovered through `numi wet-lab --help`:

1. Read `context` and retain its revision. Finish or cancel live operations and
   confirm they have stopped; a snapshot rejects changing worker output.
2. Run `numi wet-lab snapshot --revision REV`. Keep the returned snapshot ID and
   inspect its verified dependency list. A declared external dependency is not
   bundled evidence.
3. Run `numi wet-lab export --id SNAPSHOT_ID` to create the verified archive.
4. Stop the original test service before the restoration acceptance test. Run
   `numi wet-lab restore --id SNAPSHOT_ID --name RESTORED_NAME`, then
   `numi wet-lab open --name RESTORED_NAME --port PORT` in a persistent terminal.
   These identifiers are discovered outputs; do not ask the user to author paths
   or edit records.
5. Join with a fresh Codex session and read `context`. Inspect the preserved
   drafts, model bindings, authorization history and interrupted operations.
   Replay a retained prediction. Recover a stopped operation only after reviewing
   its current status; restore never restarts it automatically.

Restore verifies the manifest and uses a separate artifact location map. Never
rewrite sealed records to replace old absolute paths. Session credentials are
excluded from the export and issued afresh by the restored service. An old browser
session must reload to obtain the new credentials; a prior reveal authorization
history does not authorize revealing a different experiment.

External dependencies are listed and verified; this is not a claim that arbitrary
legacy assays are self-contained.

## Explore an admitted measured cohort before a model is available

Check the assay capabilities. A measured cellular cohort advertises
`measuredExploration: true` and `prediction: false`; it opens in the existing
population workspace with prediction disabled. Registering a compatible model is
required before inference. Never substitute a frozen cellular or spatial model
merely because that other model can execute.

The dataset preview shows source populations, conditions, intervention support and
control/outcome counts. Search interventions through bounded metadata pages. Gene
catalogs may contain only an initial page: absence from that page does not establish
that a gene is unmeasured. The gene autocomplete requests the owning dataset's
bounded feature search, and the owner validates explicitly typed genes against the
full source axis.

Select an intervention and gene to load at most 128 measured values per group on
one cell page. The plots show source-qualified cells on an expression-value axis;
vertical spacing is only for visibility. They are neither spatial positions nor
simulated trajectories. Owner-computed means remain distinct from the displayed
bounded page. More cells or pages do not become independent biological replicates.

Choose raw counts or the explicitly labeled `log1p(count / retained-panel cell total ×
10,000)` readout. This transformation is supplied by NumiVivo, not recomputed by the
browser, and is distinct from the older frozen model's log1p(CPM) values. Missing
features and inaccessible observations remain unavailable, including for reserved
targets whose metadata and counts are discoverable. Exploration never authorizes
observation reveal, objective ranking or prediction. Researcher usability remains
explicitly pending.

### Bounded measured reads from the same selection

Use `numi wet-lab catalog` to discover measured datasets as well as executable
models. `capabilities.prediction: false` is a dataset awaiting a compatible
model, not a request to substitute the spatial regression. Select the exact
specimen, population and condition with the normal revision-checked `select`.

`numi wet-lab readout --kind interventions --limit 32` lists that population's
accessible and reserved coverage. `numi wet-lab readout --kind features --search
GENE --limit 50` searches beyond the initial bounded catalog feature page.
`numi wet-lab readout --gene GENE --target TARGET --normalization raw_counts
--limit 64` reads a bounded page from the acknowledged shared selection. Use
`--normalization log1p_10000` only when requesting that explicit transformation;
its denominator is the retained source panel, not a claim of complete RNA.

Reads neither authorize reveal nor create predictions. Reserved outcomes stay
closed even when controls are accessible. Feature absence is not measured zero.
A source GEM group is a technical capture; unknown biological units remain
unknown. No researcher usability qualification is implied by automated tests.
