# Explicit-flute shipping arena audit, source 106

This is a source-bound CPU audit of the authored explicit-flute world at `/Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106`. It does not initialize Metal or claim shipping execution.

The exact authored topology is 528,444 prepared FEM nodes, 1,477,728 tetrahedra, 260,835 cohesive faces, 1,054,876 exterior faces after starch splitting, 6,335,568 rigid node/proxy pair rows, 12 proxies and 9 scene body slots. Source106’s CPU preflight reports `whole_process_preallocation_admission.complete=false`; runtime resident, Metal `currentAllocatedSize`, and host peak were not measured for this world.

## Direct admission blockers

- The shipping full-contact residual path admits at most 8,192 FEM nodes, 32,768 tetrahedra and 131,072 deformable-contact slots; its checked private arena is capped at 128 MiB. The shipping world exceeds the node cap by 64.5x and the tetrahedron cap by 45.1x (`runtime.mm:10111-10117, 10327-10368`).
- Dense private rigid contact admits at most 65,536 pair rows, versus 6,335,568 here. The sparse path uses a 65,536-candidate workspace and only admits a validated one-free-BOX six-DOF scope; it does not admit this nine-body/twelve-proxy shipping layout (`sparse_private_contact_admission.hpp:11-19, 69-108`; `runtime.mm:6106-6164, 10155-10202`).
- The non-tiled cohesive private path is bounded to 4,096 faces and 8 MiB. Ordered 8 MiB pages can process this 260,835-face set in three passes at 87,381 faces/page, but that does not waive the FEM or contact limits (`runtime.mm:5964-6048`; `cardboard_shipping_memory.hpp:915-959`).
- Selector-5 Graph Schwarz is rejected for shipping in the probe and its graph builder caps the coupon path at 4,096 nodes, 32,768 tets, 128 patches and 96 MiB (`cardboard_probe.mm:1905-1908`; `regional_graph_schwarz.hpp:91-121, 208-210, 300-323`). Its factor/original-matrix/pivot/correction buffers are allocated at `runtime.mm:6549-6569`.

## Large arenas and checked formulas

The source106 preflight’s compact-surface, 131,072-slot case models 13,796,875,626 combined bytes (14,281,570,410 with the tangent cache), below 16 GiB. It is still explicitly not admitted and not measured. The non-compact equivalent is 15,379,147,754 bytes (15,863,842,538 with cache), leaving little modeled headroom even before unknown allocator/compiler peaks. The default 8-per-tet capacity is 11,821,824 slots and its modeled compact combined total is 28,527,223,146 bytes.

Fresh C++ `sizeof` checks against source106 give 16-byte `float4`, 80-byte FEM node, 96-byte tet, 384-byte cohesive face, 96-byte cohesive force vector, 432-byte deformable contact, 48-byte contact history, 64-byte surface primitive, 16-byte contact pair, and 160-byte rigid contact sample. Independent formulas exactly reproduce source preflight values:

- Compact persistent surface: `80*S + 64*bit_ceil(S)` = 218,607,808 B. Legacy all-four-faces surface: `80*(4*T)+64*bit_ceil(4*T)` = 1,009,743,872 B.
- Candidate capacity 131,072: 624 B/slot = 81,788,928 B, plus node ranges and counters. Opt-in sorted incidence adds `12*bit_ceil(6*K)` = 12,582,912 B at that K; the legacy 8T capacity would add 1,610,612,736 B. The preflight persistent-contact formula does not include this optional sort workspace (`runtime.mm:3080-3144, 5300-5315`).
- Cohesive persistent state is 1,856 B/face = 484,109,760 B. The private page is bounded independently to 8,388,576 B at one environment.
- Persistent rigid contact is 1,579,685,376 B across all 6.3M rows. Dense private trial storage is 1,064,375,428 B; K=65,536 sparse trial storage is 36,090,192 B, before any optional parity arena.
- A complete 15-state checkpoint is 1,916,871,168 B, with two host checkpoints modeled as 3,833,742,336 B.

The full arithmetic and C++ ABI size output are in `audit.json`, `sizeof.cpp`, and `sizeof.txt`; all formula equality checks pass.

## Estimator limits and first scaling work

The shipping estimator is an extrapolation from a measured source032 runtime anchor, with coarse source/compile/staging reserves. It correctly marks admission incomplete, and does not treat its under-16-GiB scenarios as success. It does not model the selector-5 Graph Schwarz arenas; current shipping guards reject the graph mode before that can cause a false admit. It also omits the opt-in sorted-incidence key/payload workspace, so future use must add that checked extent to the preflight.

The first safe scaling step is exact sparse spatial candidates for the all-proxy rigid pair domain, retaining canonical pair IDs and accepted history and failing closed on overflow. The current sparse helper cannot be generalized by changing a limit: it qualifies only the exact one-free-BOX path. Next, the full FEM/contact residual needs a bounded streamed or tiled implementation that preserves deterministic residual accumulation and current acceptance, then a same-topology measured resident/host peak. Compact immutable surfaces, sorted node incidence, and tiled cohesive force evaluation are useful current opt-ins; none independently unlocks full shipping admission.

No cap was raised, no shell substitute was used, and no native solver step or physical qualification is claimed.

## Interpretation and reproducibility

The source106 v6 preflight already preserves the distinction between modeled byte totals and admission: `current_admission` and `known_admission_blockers` enumerate the failed gates, while `whole_process_preallocation_admission.complete` remains false. This audit does not report a false admission. Its compact byte-fit scenario is a planning estimate only; the source032 runtime-resident anchor is not the prepared shipping world, and no full-world runtime or host peak was observed.

Re-run `python3 validate_audit.py` to verify the frozen-source bindings, retained raw preflight, count-gate failures, independent byte formulas, and compiled C++ structure sizes. The package manifest binds every deliverable and the exact validation command. No Metal initialization or solver step is part of this check.
