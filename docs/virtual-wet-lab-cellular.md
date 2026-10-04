# One shared cellular and tissue laboratory

This first delivery connects an existing frozen NumiVivo cellular checkpoint to
the shared NumiLab Codex workspace. Subsequent ingestion and population-learning
work can register artifacts through this route; that scientific work has not
been performed by this integration release.

The workspace discovers owner capabilities instead of filtering for a spatial
family. Proposals retain exact specimen, population, condition, model, features
and molecular objective. NumiVivo validates the complete request. Objective
registration is durable before native inference. Unknown conditions and missing
features are blocked, retained in the draft and corrected only by an explicit edit.

The population renderer displays bounded one-gene readouts, selected-population
controls, predicted and measured population means, signed residuals and owner-
computed baseline errors. It does not invent tissue geometry or cell distributions.
Cancelled requests and stale responses cannot replace newer selections. Visual
and Codex edits use the same revision-checked cards.

## Operate it through the installed plugin

```sh
numi wet-lab context
numi wet-lab catalog
numi wet-lab select --revision REV --assay cellular-coarse-target0-v04 \
  --specimen GSE90546 --population 'GSE90546|K562' --condition K562
numi wet-lab propose --revision REV --genes FOS JUN --targets DDIT3 SPI1
```

Read the current revision before every edit or action. Human edits remain
visible and authoritative. Seal with `seal --revision REV --id CARD`; revealing
requires a separate explicitly authorized `reveal --authorized-reveal` operation.
Use `replay`, `cancel`, `recover` and `history` on the same card. `context --full`
includes full retained registration metadata; the default removes repetitive
registration bodies without changing shared state.

## Restore the laboratory

Experiments default to durable `~/.numi/laboratories`, outside disposable checkouts.
The existing filesystem store now supports:

```sh
numi wet-lab snapshot --revision REV
numi wet-lab export --id SNAPSHOT
numi wet-lab restore --id SNAPSHOT --name restored-laboratory
numi wet-lab open --name restored-laboratory
```

Snapshots reject live workers and stale workspace revisions. Legacy direct owner
writes share the snapshot lock. A manifest hashes the workspace, scientific and
UI owners, model assets and explicit external dependencies. Python package and
native-architecture requirements are recorded and checked. Exported archives can
restore after the original snapshot directory is gone. Path traversal, symlinks,
unregistered files and changed dependencies are rejected.

A separate location map resolves copied assays and owners. Sealed records remain
byte-for-byte unchanged. Restored active operations become interrupted; cancelled
operations remain stopped. Recovery requires an explicit action. Authorization
history is preserved, while the service generates fresh session credentials.
The snapshot manifest describes the immutable exported checkpoint; the restored
workspace gets a new revision and can subsequently evolve.

The cellular runtime, Metal library, weights and inputs are included. Historical
spatial source H5AD files and their retained runtime remain explicitly verified
external dependencies. This release does not claim cross-machine independence
from those legacy dependencies.

## Qualification and scientific limits

The fresh-session acceptance card is `69e92507f1144c1d84b97873a2292b34`, native
record `35a22351d1e748748ca9ec16cb4f0475`. A browser keyboard edit changed FOS/JUN
to FOS/JUNB while preserving DDIT3/SPI1. Codex rejected its stale revision,
preserved the edit, corrected an unsupported condition, sealed, authorized reveal
and replayed the exact cellular artifact. A separate spatial card completed the
same shared lifecycle. See the qualification receipts alongside this document
for restoration, recovery and exact record IDs.

The model is the existing 240-step coarse-descriptor transfer checkpoint, not
v0.3 spatial predictions. Its 27 queries and 516-feature historical intersection
are previously exposed technical development data. The K562 comparison contains
1,769 distinct controls and 477/696 DDIT3/SPI1 outcomes, with unresolved independent
biological-unit identities. This integration does not establish population-input
benefit, transferable biology or a new independent evaluation.

No intervention was the numerical choice and best observed eligible candidate
for this specific reduction objective. The UI keeps `reliableWinner: false` and
the large gene-level errors visible. No biological promotion occurred.

Local qualification includes 17 shared-state/restoration tests, 15 existing
NumiVivo laboratory tests, native exact replay/relocation/tamper checks, real-service
Chromium interaction, and three fresh Codex sessions. Browser automation and handler
checks are engineering evidence only. Safari and physical touch remain unqualified;
an uncoached researcher session remains explicitly pending.

The initial fresh session exposed a CLI `urllib` binding bug; its failed attempt
is retained and the corrected CLI was used for completion. It was not hidden as
an uninterrupted first-pass success. The Replogle–Nadig onboarding, native masked
multi-cell comparison, learned external reference and remaining UX breadth are
subsequent deliveries, not implied by these software checks.


## Published restoration artifact

Final snapshot: `32eaf6a6745244d5bfa98619b9e4fc08` (workspace revision 51).
Archive SHA256: `8a88cb011907c03480383abe56a42656ab0486513305d927e94d1ede155eca40`.
It includes the recovered but unrevealed record
`3c17bc2511a94c539558b9278c7a6eb5`. A final fresh Codex session restored this
snapshot, replayed the acceptance card and left the background service alive
at revision 55. The original archive-only restoration verified all 28 cellular
and 35 spatial sealed files byte-for-byte, with the original workspace unavailable.

The first foreground service stopped when its CLI session ended. This failure
was retained; `open` now launches a persistent local process by default, with
`--foreground` available for attached execution. The final fresh-session test
verified persistence after session exit. No measurements were revealed on recovery.

Download the snapshot from the
[cellular laboratory release](https://github.com/Numi2/numiVivo/releases/tag/virtual-wet-lab-cellular-20261004).
Legacy dependencies listed in the manifest remain required; biological promotion
and researcher usability remain unearned.
