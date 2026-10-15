# Shipping runtime capability inventory — source 076

This is a read-only, source-bound inventory for the explicit 300 × 200 × 200 mm shipping mesh. It combines frozen Matter snapshot 076 with the separate sparse-prebuild wiring overlay 075 and the CPU preflight output. The source receipt, selected source hashes, prebuild patch hash, and preflight hash are in `inputs.json`; the preflight payload is preserved in `preflight-076.json`.

**Evidence class:** exact authored cardinalities plus checked source-layout/arithmetic formulas. This is not a compiled-world Metal initialization, observed device-allocation peak, native solve, contact-capacity qualification, or physical simulation. The 16 GiB figures below are estimates only.

## Cardinalities and selected bounds

| Symbol | Meaning | Value |
|---|---|---:|
| N | prepared FEM nodes (base nodes plus duplicated glue-side nodes) | 528,444 |
| T | tetrahedra | 1,477,728 |
| C | cohesive mortar faces if both starch and HMA laws are enabled | 260,835 |
| F | compact exterior surface faces after split | 1,054,876 |
| P | eligible FEM-node × rigid-proxy pair rows (12 proxies) | 6,335,568 |
| P_upper | all prepared nodes × 12 proxies | 6,341,328 |
| K_rigid | opt-in sparse private rigid trial sample capacity | 65,536 |
| K_self | explicit deformable active-contact slot capacity | 131,072 |
| E, restart | planned environment count and FGMRES restart | 1, 16 |

The default deformable slot heuristic is `8*T = 11,821,824`; the selected 131,072 bound is a separate explicit capacity choice, not an observed bound on all contact states.

## What the gates protect

| Gate | Source purpose | Shipping result | Classification |
|---|---|---|---|
| `uint32_t` dispatch/flattened indexing | Dispatch counts and Metal element indices are U32; runtime checks `E*P`, `F`, `4*bit_ceil(F)`, and host-size multiplications before allocation. | At E=1: P=6,335,568; 4*T=5,910,912; BVH leaf count=2,097,152 and `4*leaf=8,388,608`; K_self=131,072 and `6*K_self=786,432`. All are below U32. | These checks are necessary overflow protection, but do not explain present N/T/C/K admission caps. |
| Dense private rigid trial | Dense trial samples are capped at 36 MiB and the dense private pair domain is capped at 65,536. | P exceeds the dense limit. Dense sample lower bound alone is P×160 = 1,013,690,880 bytes. | Byte/workspace cap, not U32. |
| Sparse private rigid trial | The sparse helper checks E×P U32 and a 36 MiB arena. For K=65,536: exact-P arena 36,090,192 bytes; all-node conservative P_upper arena 36,113,232 bytes. | The private-sample slice fits its cap. However, it keeps a canonical P-entry slot map and scans all P rows to classify each trial; production still evaluates and stores dense P samples. | Reduces private trial sample storage; it is not an end-to-end spatial broadphase or an O(K) contact path. |
| FEM full-contact trial | Runtime gates E≤4, N≤8,192 at encoder admission, T≤32,768, K_self≤131,072 and a private deformable budget of 128 MiB. The allocator's own N bound is 16,384, so the stricter encoder N bound also matters. | N and T exceed the scope limits by large factors. At K_self=131,072, self-contact private slots plus node ranges cost 71,369,696 bytes if persistent surface arrays are aliased, or 289,977,504 bytes if the private trial also owns the surface/BVH/sort arrays. | Mostly implementation/scratch/qualified-scope bounds. The K overflow path reports failure instead of truncating candidates. |
| Cohesive private trial | Private face+force workspace is limited to 8 MiB and C≤4,096 in the full-contact Armijo scope. | C=260,835; one face+force trial arrays at 384+96 bytes/face cost 125,200,800 bytes without aliasing. | Scratch and validated-scope bound, not device-index overflow. |
| Contact preconditioner | Selector 4 supports only the bounded one-object static/kinematic FEM-contact slice (N≤8,192, T≤32,768, C≤4,096). Selector 5 graph Schwarz is contact-free and has N≤4,096, T≤32,768, E≤4. | Neither supports this shipping topology. Modeled selector 2/4 preconditioner storage is 456,581,968 / 837,073,744 bytes, above the 256 MiB preconditioner arena budget. | Unqualified implementation and actual memory cap; no accepted contact-aware large-mesh coarse/fine solver path. |
| Proxy scope | Full-contact residual Armijo allows 2–16 proxies, with capsule/box restrictions and at most one free non-articulated six-DOF BOX in the admitted extension. | 12 proxy count passes this numeric gate, but does not establish geometry/body ownership or end-to-end solver support. | Capability-scope condition, not overflow. |

The compact surface/BVH arrays are 218,607,808 bytes at E=1: F×64-byte surface records, 4×2,097,152 float4 bounds, and four F-sized U32 sort arrays. This is within U32 address space and is counted by the CPU layout proof.

The source preflight's bounded `K_self=131,072`, compact-surface, sparse `K_rigid=65,536` scenario models 12,424,781,782 combined bytes (and an alternative best case 13,796,875,626 with different rigid storage). The same preflight marks the full private arenas **rejected** by runtime limits and says the model is not a measured peak. A separate compact-surface, K_self=131,072 scenario is modeled at 13,796,875,626 bytes. Do not read either “within 16 GiB” result as actual device admission. At default K_self=8T, the modeled combined allocation exceeds 30 GB.

## Work behavior and first scaling repair

Rigid contact is a node×proxy domain, not an all-node-pairs self-contact graph. Twelve proxies make P linear in N; the main costs are dense persistent samples (one P×160-byte array is already 1.014 GB), canonical pair/history storage, and repeated per-trial classification. The opt-in sparse path compacts trial samples to K, but `nm_contact_build_armijo_sparse_pair_flags` writes/checks all E×P entries, its 32-lane scan walks the full P table, and `nm_contact_compact_active_sparse` initializes P-sized maps. It also does not remove production dense P sample evaluation. This is a memory slice, not a complete rigid broadphase.

Self-contact already has a conservative exterior-surface Morton/BVH route; it does not unconditionally enumerate F². It nevertheless rebuilds the surface records, sorts them, constructs a 2,097,152-leaf BVH, counts/query-scans/scatters candidate pairs on each private trial. The count kernel asks the BVH for each active left face; broadphase overlap can still approach quadratic work in a dense-overlap case. Candidate count greater than K_self is explicitly an overflow failure. The current count scan uses one 32-lane threadgroup per environment and walks left-face counts in 32-entry chunks; at F≈1.05M this serial scan stage itself becomes a scaling hotspot.

**First mathematically equivalent scaling slice before lifting the current caps:** make the existing counted self-contact pipeline streamable without losing candidates. Keep conservative swept AABB/BVH coverage and canonical source-face pair identities; tile left-face counting/scatter and consume candidate/contact/incidence pages in deterministic ID order, with a global checked count and fail-closed byte budget. Persist/rollback friction history by stable canonical pair identity; do not drop off-page candidates, truncate overflow, or reuse the 131,072 slot limit as a physical claim. This first addresses the large per-trial surface/BVH/contact workspace while preserving the existing broadphase predicate. A fixed-tree refit could reduce the repeated sort/build cost, but needs a no-miss proof under the full candidate sweep and identity-order/rollback checks before adoption.

After that slice, rigid trial flags still need a true spatial candidate builder if P-wide classification is too costly; production sample/history storage must become sparse while retaining stable identity. Cohesive forces can be tiled over faces, but all three quadrature histories, activation references, accepted rollback, and incidence sums must be preserved; the 8 MiB private cap cannot simply be raised. FGMRES needs an admitted large-mesh contact/cohesive preconditioner; current selector 4 is small and selector 5 forbids contacts and proxies. These are separate gates, not solved by prebuild memory estimates.

## CPU proof and reproducibility

`shipping_runtime_layout_proof.cpp` includes the frozen ABI and sparse-admission helper, statically verifies the relevant record sizes, and checks pair counts, BVH leaf/index extents, sparse arena bytes, self-contact private arena bytes, and cohesive per-face private bytes. It exits nonzero if any expected formula or ABI size changes. Build/run command:

```sh
clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076/matter/include \
  /Users/home/cardboard-implementation-20261014/analysis/shipping-runtime-capability-inventory-076/shipping_runtime_layout_proof.cpp \
  -o /tmp/cardboard_shipping_runtime_layout_proof_076
/tmp/cardboard_shipping_runtime_layout_proof_076
```

The captured output is `shipping_runtime_layout_proof-076.json`. The source preflight and 075 prebuild overlay are retained as separate inputs. No files in the shared source tree were changed; no CMake or GPU test was run.
