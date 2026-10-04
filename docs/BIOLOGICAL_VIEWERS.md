# Numi biological viewers

Numi routes its retained biological artifacts into OpenAI's three native Codex
viewers. The plugins remain separately installed and updated; Numi keeps its
existing native computation and observation authority.

| Surface | Artifacts | Native plugin |
| --- | --- | --- |
| Molecular inspection | Coordinates, small molecules, topology-bound trajectories, maps | Molecular Structure Viewer |
| Sequence inspection | Sequences, alignments, annotated constructs, traces | Biological Sequence & Alignment Viewer |
| Tissue inspection | Microscopy, slides, supported DICOM and AnnData | Slide Viewer |

## User flow

Ask Numi to inspect an existing molecular, sequence or tissue result. The Numi
skill resolves the owner's artifact, loads the companion skill, opens its native
viewer and retains the same session for subsequent controls and analyses.

The Virtual Wet Lab's **Inspect biology** button copies a request containing the
current assay, specimen, population, condition, gene and revision for the Numi
conversation. This browser-to-Codex handoff is explicit; it does not automatically
open an app or embed another viewer. Numi resolves source association before
opening. A gene symbol alone cannot establish species, isoform, residue mapping,
cell identity or image registration.

`numi view catalog` describes supported primary files and identity contracts.
`numi view status` checks installed/enabled versions. Install requested companions
with `codex plugin add structure-viewer@openai-curated-remote --json` and the
corresponding `sequence-viewer` and `slide-viewer` IDs. A fresh Codex session loads
new companion tools.

## Source-bound handoff

```sh
numi view prepare /absolute/owner/output.pdb --owner NumiVivo --output inspection.json
numi view verify inspection.json
```

Preparation hashes an explicitly selected file, emits actual native opening-tool
arguments and records `viewerReady: false`. It does not parse the scientific
format or invoke a viewer. Codex calls the installed tool, retains its returned
session and verifies readiness through the plugin's current live context. Retrying
a structure opening reuses the recorded open-intent UUID.

`--companion` retains exact trajectory/map/overlay identity for a subsequent
source-admitted import into the same viewer. Topology, atom ordering and image
registration still require owner evidence. The default hashing budget is 512 MiB
per file; use `--max-source-bytes` for an explicitly larger source. No source is
modified; saving a handoff never overwrites an existing file.

## Shared selection and return

```sh
numi view context
numi view prepare /absolute/owner/output.fasta --wet-lab --revision REV --output inspection.json
numi view return inspection.json --gene EXACT_OWNER_GENE --revision REV
```

The return operation rechecks source hashes, active laboratory, selection and
revision, then uses the existing shared-selection compare-and-swap. It preserves
all other selection axes and never edits an objective or runs/reveals an
experiment. A human edit requires reviewing the changed intent and preparing a
new handoff. Resolve the returned gene from live viewer and owner identifiers;
the helper does not manufacture a biological correspondence.

Never use a full raw H5AD as a shortcut around reserved observations. The existing
owner must supply an authorized view/export; otherwise keep the bounded Wet Lab
readout. Context contains selection metadata only, with no credentials, card
observations or raw assay paths. Coordinate/sequence/slide exports retain the
viewer provenance and are admitted back into native owners through their normal
format and identity checks.

For isolated previews, start `virtual_wet_lab.py` with its own `--workspace`,
`--port` and `--connection-file`. Set `NUMI_WET_LAB_CONNECTION` to that discovery
file for `numi view` commands. The normal active laboratory remains selected.

## Verification performed on 2026-10-04

- Installed/enabled: structure 0.1.90, sequence 0.1.43, slide 0.1.65; each installed
  manifest identifies OpenAI as author.
- Inspected the actual local MCP `tools/list` opening schemas. All six
  kind/placement combinations produce compatible opening arguments. No synthetic
  host capabilities were supplied and no source was opened through the probe.
- Seven focused tests cover routing, primary/companion distinction, changed
  source/dispatch rejection, byte limits, symlinks, non-overwriting output,
  context filtering and stale/cross-laboratory return rejection.
- Existing nine Numi skill contract tests and plugin manifest validation pass.
- An isolated instance of the actual cellular cohort accepted a gene-selection
  round trip, retained unchanged cards/operations and rejected a stale return.
  Its structure source was a bundled synthetic parser fixture, not biological
  evidence or an asserted mapping to the selected gene.
- Browser inspection verified the new panel, current selection, three viewer
  choices and successful copy-request feedback with no browser console errors.

Native mounted-viewer rendering/control remains a separate fresh-session check:
the companion tools were installed after this conversation started. These results
establish Numi integration behavior, not scientific or uncoached usability
qualification. Cross-view biological mappings and whole-assay native analyses
are only available where their owning tools provide actual evidence.
