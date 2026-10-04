# Virtual Wet Lab: shared tissue investigation

Use the existing `numi wet-lab` workspace and NumiVivo owners. This is a local,
shared experiment surface, not a separate assistant, simulator, or hosted queue.

1. Read `numi wet-lab context`. The active workspace is discovered automatically;
   do not ask for assay paths or JSON. If it is unavailable, start `numi wet-lab`
   in a persistent local terminal, then open its printed local URL. The user
   selects a population on the tissue. Context returns explicit assay, specimen,
   population, experiment identifiers and a workspace revision.
2. Turn the user's molecular goal into the exact gene list they requested. Use
   `numi wet-lab propose --revision REV --genes GENE ... --targets TARGET ...
   --title 'Exact molecular objective'`. Supported targets and measurement
   coverage appear on the editable card in the same canvas workspace. An absent
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
