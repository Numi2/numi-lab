# Native child identity integration 219

## Scope and result

This is an opt-in exact identity re-encoding path for the already registered skin/fat parent. It imports TetGen node/element tables and five-column semantic labels, checks those rows against the live parent, and preserves app-owned exact F64 current/reference arrays, registered surfaces, frame maps, source ancestry, parent-tet maps, and facet/head-wall maps. The existing unpaired identity preflight/replacement path remains responsible for topology, contact, mass debit, and source-preservation checks.

Native preparation control 003 passed the actual import and install path: 7,233 nodes, 28,632 tets, 7,233/7,233 capture-current rows matched exact F32 emission of live current, 7,233/7,233 reference rows matched exact live F64 reference. Existing identity remap and identity install passed; maximum source mass delta was 1.51e-14 kg and maximum local first-moment delta 1.21e-15 kg m. The run reports zero advanced physical time and no new skin admission. See native-preparation-control-003-pass/.

The child importer requires NUMI_HUMAN_PHYSICAL_SKIN_CHILD_IDENTITY_IMPORT=1 and NUMI_HUMAN_UNPAIRED_IDENTITY_PREFLIGHT=1. Identity installation additionally requires NUMI_HUMAN_UNPAIRED_IDENTITY_INSTALL=1. The imported rows only validate identity; installation retains identityTopology and its exact existing ancestry and mass/debit checks.

## Measured limit

newmesh=false. Validation requires imported node, tet, material, donor, stable-face, and region rows to match the unchanged live parent. The available reference import is not a changed-topology child mesh. No complete changed-child O/I/D surfaces, child boundary-node maps, per-node donor-frame and source-embedding maps, parent-tet barycentric/residual ancestry, or per-cell surface-facet/head-wall maps exist in this package. No changed geometry, candidate033 admission, physical step, or new skin qualification is claimed.

The capture format is representation-specific: current_* is compiled Float32 while rest_* is Float64. Captured current must equal the exact IEEE-F32 cast of live current; captured rest/reference must equal exact live F64 reference; TetGen node rows must equal exact F32 emission of live reference. After those comparisons the runtime bundle retains exact live F64 current/reference. There is no geometric tolerance or rounding of owner state.

## Retained failed controls

Control 001 failed because its first coordinate comparison required exact F64 equality for both capture columns. Control 002 passed exact table representation, then failed in a changed-child spatial-mass mapper that was unnecessary for identity topology and required exact F32 owner coordinates. The current identity-only path skips that mapper. Both failed attempts remain preserved beside the passing control 003 receipt.

## Changed-child native contract still missing

The current TetGen parser imports only nodes, tets, and semantic labels. The identity initializer copies parent O/I/D surfaces and topology sidecars, and the validator intentionally rejects any child whose node/cell/label rows differ from the current parent. A changed child therefore needs a separate complete child-sidecar import path into PhysicalSkinChildMassMappingBundle: three RegisteredSurface geometries with per-row donor frame/source embedding; an O/I/D surface-to-volume-node boundary map; per-volume-node donor owner/local transform/origin plus source vertex/face/barycentric/residual and parent-tet/barycentric/residual ancestry; child cell material/donor/region/stable-source identities; four facet-to-surface/source-face rows per tet; and explicit head-sidewall flags/identities. Those records must be tied to the emitted child node/tet hashes and validated before calling the existing conservative mass mapper and append/replacement owner. The present identity path cannot synthesize these maps from .node/.ele and the five-column labels CSV.

## Publication staging

Staging was built in external index P/native-child-identity-integration-219/publication-957b85-staging.index, based on main commit 957b85f9cc57cefe2dc182e49e1e90426e23737. The exact three-file list and staged patch are retained here. The shared worktree HEAD and real index were not modified by this staging. The target diff check passed. Root owns publication; no push or commit was performed by this worker.
