# Six whole-body serialization replacements, SSH Numi, 2026-11-02

Source published as 3746de2401f7c3a52855a2f0eb7ff89bf29a9294 by numi2. This increment changes
execution algorithms inside the existing Matter/Human owners. Geometry,
anatomical identities, material equations, timestep, mass mapping, contact
admission and accepted-state rollback are unchanged.

| Path and fixture | Original median ms | New median ms | Median paired speedup |
| --- | ---: | ---: | ---: |
| Node validation, 318,974 nodes | 163.210 | 0.457 | 356.84 |
| All-node force audit, 318,974 nodes | 56.686 | 0.155 | 365.54 |
| Active reaction audit, 318,974 nodes | 45.271 | 0.189 | 237.27 |
| Anchor projection, 128 DOFs, 18,765 active of 318,974 nodes | 49.235 | 12.127 | 4.06 |
| Support working-set pivot/reset, 318,974 nodes | 4.623 | 0.284 | 16.27 |
| Vascular working-set pivot/reset, 318,974 nodes | 4.535 | 0.340 | 13.44 |
| Topology, 8,193 nodes / 24,579 tets | 13.627 | 0.215 | 63.27 |
| Blood compaction, 1,024 particles / eight requested merge slots | 500.880 | 13.398 | 37.61 |

Production shaders compiled from the publication checkout; three matched GPU
fixture runs, original then replacement, fixed order. GPU start/end times cover
complete command buffers without counter sampling or encoder splitting.
These are fixtures, not timings of those paths inside the full Human.
Ratios are individual measured gains and must not be multiplied.
Most fixtures use two environments; deferred anchor projection uses one.
performance-summary.json retains all three values and ratio ranges.

Algorithm changes:
- Node validation evaluates one row per lane with deterministic minimum
  invalid-row witnesses and a hierarchical finish. Dynamic transfers and
  current poses remain validated each attempted step.
- Force/reaction audits use 256-row tiles and hierarchical finite-aware sums.
- Immutable, source-order active-anchor indices replace per-DOF whole-node
  scans in both projection owners. Point Jacobians and force equations remain
  unchanged. Real active anchors are still evaluated independently per DOF.
- Vascular and support working-set decisions retain their original small
  pivots. A GPU-generated indirect count suppresses no-pivot whole-vector
  clears. Active clears visit all FEM, field, rigid, MPM, support and vascular
  rows in parallel, preserving the finite-value failure rule and barriers.
- Topology emits actual tet corners, scans node counts hierarchically,
  scatters into existing incidence storage, then sorts each actual adjacency.
  Source-order mass summation and incidence match bit for bit. This avoids a
  duplicate full corner-sort arena. Mutation, erosion and growth encoders use
  the same replacement; current Human topology is conditional.
- Blood compaction uses a rebuilt spatial hash after each changed COM,
  evaluates local candidates cooperatively and reduces exact distance/A/B
  ties. Original nearest-pair order, merge equations, material-history tests,
  mass/volume/momentum/angular momentum and dissipation ledger are retained.
  Extreme coordinates use a parallel exact-search fallback. Sequential merge
  publication remains intentional. Current Human has blood MPM disabled.

Acceptance evidence:
- Production native build succeeds with Metal4 -O3 -fno-fast-math.
- Publication adapter and runtime syntax checks pass.
- Three combined GPU parity runs pass, including tile edges, late faults,
  both environments, duplicate tet corners, high valence, every coupled clear
  block, no-pivot indirect suppression, material incompatibility, spin,
  capacity failure, empty particles and extreme-coordinate pair fallback.
- Topology/mass, sparse projection, working-set state and blood
  particle/history/ledger comparisons are bitwise. Reordered force audits
  pass the retained 1% plus 1e-6 absolute gate.
- Existing native attachment, deformation, rejected-step rollback and replay
  controls pass with the new node/anchor/working-set/topology paths forced on.

Retained limits and negative results:
- No-pivot support work is 0.0070 ms versus 0.0044 ms; metadata/dispatch
  overhead is measurable. No-compaction blood work is 2.152 ms versus
  1.788 ms. Tiny workloads may favor the retained original paths.
- Pathological high-valence topology is about 4.32x faster, below the regular
  fixture's 63.27x; local adjacency sort/mass work is still serial per node.
- The legacy provisional validator could overwrite duplicate/invalid index
  rejection with a successful transfer lookup. Its admission flag is now
  guarded; static CPU admission already rejects those malformed sources.
- Earlier compiler and test failures are retained. controls/matched-0
  stopped on a CPU copy from an empty synthetic particle vector; the fixture
  copy was corrected. It was not a production GPU or physical failure.
- native-import-122 stopped after 50.254 seconds under the unchanged
  disk guard (below 12 GiB), before any physical step. It does not establish
  native motion, solver convergence, accepted skin/fat transfer or 0.1x.
  The OS retained about 16 GiB swap; no OS swap or protected source was deleted.
- Full-scene preparation/working-set memory admission remains a separate
  blocker. No restart request, relaxed guard, hidden assistance or acceptance
  bypass was used to claim performance.

Reproduce with tests/human_body_parallel_work/run.py and the exact library
and source identities in publication/controls-002/run-receipt.json.
All execution was on SSH Numi. The native122 plan retains exact input hashes,
geometry/mass gates, command, environment and resource monitoring.

Storage follow-up: a lossless compression attempt on the shared 840 MB
immutable preparation dump produced an APFS clone without allocation gain.
It was not promoted; the temporary clone was removed and all 12 original
hardlink paths remain untouched. The existing hash-pinned dump is referenced
instead of duplicated in this evidence. No protected input or OS swap was
deleted. Disk remains below the native qualification reserve after OS paging.
