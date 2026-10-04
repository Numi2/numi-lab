# Measured cellular onboarding through the shared laboratory

The Replogle–Nadig engineering cohort is usable in the existing NumiLab Codex
workspace. Its count stores and biological access policy belong to NumiVivo;
NumiLab provides the population view, editable proposals and execution state.
There is no compatible trained model for this new cohort yet. Prediction stays
blocked, with the objective retained and a specific model-registration correction.
Existing frozen cellular and spatial records remain unchanged.

| Source/context | Admitted cells | Distinct controls | Supported targets |
|---|---:|---:|---:|
| Nadig HepG2 | 6,944 | 512 | 201 |
| Nadig Jurkat | 8,128 | 512 | 238 |
| Replogle K562 | 7,232 | 512 | 210 |
| Replogle RPE1 | 7,840 | 512 | 229 |

The cohort totals 30,144 cells and a 256-target vocabulary, selected by frozen
metadata support and deterministic ordering. Source gene panels map to 11,907
feature IDs with explicit missing-feature masks. These numbers are not numbers
of independent biological units or successful unseen-target predictions.
Biological-unit independence is unresolved; GEM groups are technical captures.
Known-target, held-target and held-context development assignments stay separate.
Only the intersection of their permitted training outcomes is open in the UI;
controls are explicitly accessible references. RPE1 outcomes remain reserved.

## Investigate measured evidence

Open the cohort in the existing assay selector. Choose its population, search
for a measured gene, and select an intervention. Metadata is paginated and gene
reads return at most 128 control and 128 accessible treated values per page.
Raw UMI counts and `log1p(count / retained-panel cell total × 10000)` are distinct.
The browser renders owner-computed means; it does not implement another scorer.
Missing features remain unavailable, and partially reserved groups display only
accessible values. Neither exploration nor a proposal authorizes reveal.

Codex uses the installed plugin's normal shared operations:

```
numi wet-lab catalog
numi wet-lab context
numi wet-lab select --assay ASSAY --specimen SPECIMEN --population POPULATION --condition CONDITION --gene NOC2L --target AARS --revision REVISION
numi wet-lab readout --kind features --search NOC2L --limit 50
numi wet-lab readout --kind interventions --limit 32
numi wet-lab readout --gene NOC2L --target AARS --normalization raw_counts --limit 64
```

Use identifiers discovered from the catalog and the latest revision. Draft edits
still use compare-and-swap; human edits cannot be overwritten by stale commands.
A fresh Codex session discovered the cohort, inspected 64 of 512 controls and all
22 accessible AARS outcomes in HepG2, then created the visible NOC2L proposal
`842cb49d7ea147398b06fbde4fa2580b`. The ten reserved AARS outcomes stayed closed.
No model substitution, inference, sealing or reveal occurred in that session.

## Qualification and limits

NumiVivo verified all three four-source composite corpora and reconstructed 64
sampled original-source rows exactly from native counts. Access tests check
reserved rows, actual cross-panel missing genes, tiles, invalid conditions,
prediction blocking and deterministic metadata-only sampling. Seventeen shared-state, model-comparison
and snapshot tests pass. Twelve synthetic browser checks and the actual-cohort
Chrome workflow pass, including obsolete-response cancellation. Real testing
found and fixed a stale readout during population changes and misleading units
on unexecuted proposals. Individual gene replies stayed below 16 KB.

The first fresh-browser/cold-service run took 4.48 seconds to metadata readiness.
It missed the subsequently declared exploratory 2-second target. Deferring
expression verification until the first requested source read reduced a later
cold-service run to 3.00 seconds for metadata and 3.40 seconds for the first gene
reply, still missing the metadata target. That rerun had a concurrent access
qualifier; OS/disk caches were not cleared in either run. These are scoped local
engineering measurements, not general Apple hardware or full usability evidence.
Source metadata may render before count-file verification; each requested source's
expression artifacts are hash-verified before returning measured values.

Safari and physical touch remain unqualified. Researcher usability is explicitly
pending at the user's direction. No biological promotion or new training is
claimed. Full eligible-source admission, the native mean-versus-population input
experiment, learned comparison and compatible model registration remain later
deliveries. The full-source parameter path is implemented but has not been
executed; the engineering subset must not be described as full admission.

## Restoration executed

Snapshot `5c885e23cc5145bc9a71a78f4df9d12f` was exported, then restored as
`cellular-cohort-v04` after stopping the original service and making the original
workspace, cohort and snapshot paths unavailable. All 90 retained experiment
files matched their snapshot hashes before replay. A fresh Codex session found
the measured proposal, read the same 512-control / 22-accessible-outcome HepG2
comparison, and completed exact native replay of the previously revealed
cellular experiment. The other sealed experiment remained unrevealed. Native
composite verification passed using a new location map without rewriting the
scientific records. Historical spatial sources and the native runtime remain
explicit verified external dependencies.

The initial 996 MB export finished after its CLI connection timed out at 120
seconds; the result was recovered and verified. Snapshot/export requests now
allow 30 minutes, and new archives use faster lossless gzip compression. No
scientific artifacts or sealed records change with the compression setting.
The final UI also makes total, accessible and reserved outcome counts explicit.
The real restored-data browser flow passed after that presentation correction.

See [retained qualification evidence](evidence/cellular-onboarding/) and the
[NumiVivo source-bound cohort contract](https://github.com/Numi2/numiVivo/blob/main/Tools/VirtualWetLab/CELLULAR_COHORT.md).
