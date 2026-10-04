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

For an admitted measured cellular cohort, **Prepare selected cells for Slide
Viewer** creates an actual AnnData source from the current gene/intervention.
It exports at most 128 accessible cells per group, preserving original cell IDs
and owner-supplied raw UMI counts. It neither opens the raw assay nor creates
spatial coordinates. The copied request then names the verified artifact rather
than asking Codex to find an unspecified source. Unsupported assay families
retain their normal owner workflow.

The same path-free operation is available to Codex:

```sh
numi view context
numi view selected --revision REV
```

The result includes a retained handoff, dataset/access fingerprints and explicit
page coverage. Optional `--limit` (1–128) and `--offset` select a bounded page.
The existing owner admits observations before serialization; reserved responses,
absent/ambiguous genes, identity mismatches and stale revisions fail explicitly.
No preprocessing, means, predictions or new scientific quantities are computed.
The service's existing h5py/NumPy environment writes the interchange file; these
packages are not required by the `numi view` client. Output uses a new private
directory published only after the artifact, provenance and handoff are ready.

`numi view catalog` describes supported primary files and identity contracts.
`numi view status` checks installed/enabled versions. Install requested companions
with `codex plugin add structure-viewer@openai-curated-remote --json` and the
corresponding `sequence-viewer` and `slide-viewer` IDs. Check current tool availability after installation; a fresh Codex session
loads companions when the running conversation has not refreshed them.

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
- Three additional tests verify selected-cell admission, identity handling and
  AnnData serialization in the owner's h5py environment. A real NOC2L/AARS cohort
  export contained 150 cells and one gene: every cell ID and raw count matched the
  owner readout, with no spatial coordinates and unchanged workspace revision.
  The browser prepared this artifact and copied its exact handoff successfully.

The installed tools subsequently became callable in the same conversation. All
three real opening calls created native sessions: a synthetic structure fixture,
a synthetic DNA fixture and NumiVivo's retained 4-by-3 HIRISA AnnData reader
fixture. The last has PCA coordinates but no spatial coordinates, so no tissue
image or cell position is implied. All three cards reported awaiting mount; native
rendering/control remains pending until the host mounts those cards. These results
establish Numi integration behavior, not scientific or uncoached usability
qualification. Cross-view biological mappings and whole-assay native analyses
are only available where their owning tools provide actual evidence.
