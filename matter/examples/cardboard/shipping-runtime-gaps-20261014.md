# Shipping-size runtime admission and memory gaps

Date: 2026-10-15
Scope: audit of the native shipping runtime and its sizing helpers, plus the admission-consistency repair and source-bound Fpeak numerical-capacity gate. No production solver, contact law, tolerance, mesh, or certificate behavior was changed. No GPU transaction was run for this audit.

## Effective native full-contact scope

The shipping mesh is an explicit flute/liner/medium/glue FEM mesh with 453,378 nodes and 1,477,728 tetrahedra. The full-contact residual-descent selector is currently a bounded experimental path, not a large-mesh general solver. Its transaction scope is enforced in `matter/src/runtime.mm`:

| Limit | Effective gate | Owning source |
|---|---:|---|
| Environments | 4 | `matter/src/runtime.mm:8059-8061` |
| FEM nodes | **8,192** | `matter/src/runtime.mm:8062-8068` through the capsule/box scope called by full-contact admission |
| Tetrahedra | 32,768 | `matter/src/runtime.mm:8065-8068` |
| Deformable-contact slots | 131,072 | `matter/src/runtime.mm:8275-8277` |
| Rigid contact pairs | 65,536 | `matter/src/runtime.mm:8106-8116` |
| Rigid proxies | 2–16 | `matter/src/runtime.mm:8112-8126` |
| Mixed cohesive faces | 1–4,096 | `matter/src/runtime.mm:8251-8257` |
| Private deformable-contact trial arena | 128 MiB | `matter/src/runtime.mm:4550-4585` |
| Private rigid-contact trial sample arena | 36 MiB | `matter/src/runtime.mm:4525-4535`, `8106-8126` |
| Private cohesive trial arena | 8 MiB | `matter/src/runtime.mm:4492-4515` |

The path also requires one static explicit FEM object, contact enabled, no adaptive/mutable/multiphysics or MPM path, no articulated coordinates, and at most one six-DOF free box. It admits prescribed capsule tools and static box tools, with the free-box extension available for the single body case. Requests must use one physics substep at microtick zero and omit prepared-state, adaptive-transfer, mutation, external FEM-force, and rigid-contact-constraint features (`matter/src/runtime.mm:8032-8142`). Full-contact activation additionally requires the bounded mixed-cohesive and private deformable arenas (`matter/src/runtime.mm:8245-8278`).

Before this repair, the shipping probe and memory helper admitted up to 16,384 nodes, while the runtime transaction scope rejects above 8,192. This was a false-admission window. Both shipping checks now use the shared `kShippingFullContactArmijoNodeLimit` value of 8,192 in `matter/tools/cardboard_probe.mm` and `matter/tools/cardboard_shipping_memory.hpp`. The existing memory helper check admits exactly 8,192 and rejects 8,193 and 16,384. This aligns the report with the runtime; it does not make the 453,378-node mesh admissible.

## Contact cardinality and unsafe scaling

The compiler creates an authored rigid-contact pair for every contact-eligible continuum node and every rigid proxy (`matter/src/compiler_impl.cpp:3183-3211`). The prebuild estimator deliberately counts every FEM node against every proxy (`matter/tools/cardboard_shipping_memory.hpp:724-730`). With 453,378 nodes:

| Proxy count | Potential node/proxy pairs | Compared with 65,536 cap |
|---:|---:|---:|
| 2 | 906,756 | 13.8× |
| 16 | 7,254,048 | 110.7× |
| **12 current shipping proxies** | **5,440,536** | **83.0×** |
| 39 historical shipping reserve | 17,681,742 | 269.8× |

The current shipping scene uses twelve proxies: three reusable scoring families, each with one punch and two rectangular rails, plus compression platen, floor and finite loading body. Every crease still requires its own physical indentation, withdrawal and release; family tools relocate through native accepted translation commands. The authored scene fits the16-proxy scope; source48 prebuild accounting still used39proxies/5bodies, corrected by source49shared12proxy/9body constants. Its conservative5,440,536 dense node/proxy rows still exceed the65,536-pair cap by83 times. Even the two-proxy lower case exceeds the cap by almost fourteen times. The historical39-proxy allocation calculation below is retained for comparison. The prebuild estimate’s per-pair shared cost is `sizeof(NMContactPairGPU)+16` bytes, and the runtime private trial sample alone is 160 bytes per pair; 17,681,742 pairs imply about 566 MB for that shared pair estimate and 2.83 GB for trial samples, before active maps/incidence. Runtime limits the trial samples to 36 MiB and 65,536 pairs. A smaller deformable-contact slot count cannot solve this independent rigid-pair scaling problem.

The deformable contact capacity defaults to `8 * tetrahedronCount` (`matter/src/compiler_impl.cpp:3436-3456`), which is 11,821,824 slots for this mesh. At 624 bytes per persistent slot, that arena is 7,376,818,176 bytes. The probe already exposes an explicit capacity override (`matter/tools/cardboard_probe.mm:93-94,1014-1015,1951-1953`), and kernels report `NM_STATUS_CAPACITY_OVERFLOW` if candidate or active contact counts exceed capacity rather than silently truncating (`matter/src/metal/contact.metalinc:1887-1905,4937-4982`). Consequently, choosing a lower value is physics-preserving only if it covers every reachable candidate set; an overflow must remain a rejected transaction.

For the legacy all-four-faces-per-tet surface domain, `P=4T=5,910,912` and the persistent surface/sort/BVH estimate is 1,009,743,872 bytes. With the default capacity, the estimated private full-contact arena is another 6,684,219,392 bytes: `80*P + 64*bit_ceil(P) + 480*C`. Compact immutable surface maps can replace `4*T` by the exact compiled exterior-face count and are restricted to explicit immutable FEM without mutation/spare topology (`matter/src/compiler_impl.cpp:783-800,2329-2349`; runtime primitive selection is at `matter/src/runtime.mm:54-67`). This reduces surface storage but does not remove the slot, rigid-pair, node/tet, or cohesive limits.

## Measured versus estimated memory

The only same-size native allocation anchor found is case032:

`matter/examples/cardboard/evidence/native-implementation-20261014/native/shipping-memory-preflight-032/output/shipping-memory-native-preflight.json`

It records 453,378 nodes, 1,477,728 tetrahedra, zero cohesive faces, and phase `runtime_initialized_no_step_or_native_run`:

| Observation | Bytes | Interpretation |
|---|---:|---|
| Matter runtime resident | 10,467,551,760 | Measured initialization allocation; no physics transaction |
| Metal current allocated | 10,504,568,832 | Measured device allocation at initialization |
| Process physical footprint | 5,616,488,336 | Separate unified-memory process observation |
| Metal recommended max working set | 19,069,665,280 | Advisory device value, not current allocated bytes |

The report marks the measured domains under the 16 GiB componentwise gate and warns that host/device counters may overlap. They must not be summed as disjoint memory. The prebuild static screen’s 821,124,688 bytes is a lower bound, not a complete budget. Case032 is also not a measurement of the complete shipping tool set, cohesive faces, or Armijo trial allocations.

For orientation only, adding the default-capacity private deformable arena to the case032 runtime-resident measurement gives 17,151,771,152 bytes, about 15.97 GiB, before completing the real rigid/tool and cohesive allocations. This is an extrapolation, not a measured simultaneous peak. The current twelve-proxy shipping prebuild implies rigid pair/sample costs that fail the existing caps. At160 bytes per pair, its private rigid samples alone would require870,485,760 bytes, before incidence, dense stable histories or FEM/self-contact arenas. The shipping prebuild uses `T/5` as a cohesive-face reserve when a cohesive law is requested (`matter/tools/cardboard_probe.mm:4257-4260`); that is a planning heuristic, not a proven upper bound for the compiled flute interfaces. Admission must use the exact compiled face count before runtime initialization and reject if the trial arena does not fit.

Reducing deformable capacity to 131,072 would reduce persistent contact slots to 81,788,928 bytes and private contact slots to 62,914,560 bytes, instead of 7.38 GB and 5.67 GB respectively. That makes the memory envelope more promising, especially with a compact exterior surface map. It remains an estimate: no exact active-manifold upper bound for the shipping fold/closure paths has been measured, and the current node/tet, rigid-pair, proxy, cohesive-face, and private-arena caps still reject the mesh.

## Regional preconditioners are not an escape hatch

Regional Galerkin selectors 2 and 3 require one immutable, explicit, non-mixed FEM object and exclude self-contact, rigid proxies and cohesive interfaces. The implemented selector 4 extends the bounded operator to prescribed tooling, self-contact and mixed cohesion, with independent native AP/RAP and nodal-block checks in source46 case090. It still limits the mesh to 8,192 nodes/32,768 tetrahedra, uses a 256 MiB coarse arena, and excludes free-body/generalized coupling. This is a conditioning change on the same physical operator; it does not admit the shipping mesh or establish complete-cycle convergence. Scalar diagonal remains the compatible option outside that bounded selector's scope.

## Minimal staged path without changing the physics or certificates

1. **Reduce the actual contact domain conservatively.** Keep all explicit flutes and enable the immutable exterior-face map. Replace the static all-node × all-proxy pair table with per-step spatially bounded FEM/tool candidates over conservative swept tool bounds. Include every pair that could become active, preserve stable pair/history identity, and validate against the current all-pairs method on bounded meshes.
2. **Bound deformable contact state transactionally.** Establish a conservative upper bound for candidate contacts over scoring, folding, closure, compression and drop. Either reserve that bound or implement grow-and-remap before a transaction commits. Preserve accepted friction history and return an overflow rejection if the bound is exceeded; never truncate or reset history to fit memory.
3. **Stream private trial work without changing the acceptance functional.** Tile contact and cohesive trial evaluation to bound scratch. Every tile must contribute to the same deterministic full residual and Armijo objective; candidate material/contact/cohesive histories remain isolated until one global acceptance, then commit once. Existing candidate/sort/BVH/incidence buffers are mutable, so sharing them between trials is unsafe without an explicit rebuild/lifetime proof.
4. **Generalize count and byte limits only after bounded evidence.** Rework the 8,192-node/32,768-tet and 128 MiB admission limits with checked 64-bit sizing, typed dispatch/index limits, allocation-failure rollback, and tested overflow cases. Keep solver tolerances, residual definition, and certificate gates unchanged.
5. **Re-measure the full compiled world before stepping.** Include exact exterior-face count, all 12 score workflows with reusable tool proxies, exact rigid pair count, exact cohesive face count, private residual-descent arenas, snapshot/checkpoint overlap and host/device high-water. Then run bounded contact/overflow parity and solver-certificate regression before a shipping-scale step.

Relevant implementation owners/paths are `matter/src/compiler_impl.cpp` for pair/surface topology, `matter/src/metal/contact.metalinc` for candidate capacity and overflow, `matter/src/runtime.mm` for private Armijo allocation/admission and transaction semantics, `matter/tools/cardboard_shipping_memory.hpp` for sizing, and `matter/tools/cardboard_probe.mm` for shipping admission and evidence. Current result: the admission consistency bug is fixed, but there is no current native path that executes the full shipping mesh with the required complete contact/cohesive residual-descent semantics under the declared 16 GiB limit.

## Fpeak masses require spatial and temporal evidence

The first-local-peak detector and nominal-versus-half-speed comparison remain useful measurements, but rate sensitivity is not mesh or timestep refinement. The probe now writes the exact accepted nominal branch to `shipping-compression-nominal-curve.csv`. It continues to export complete compression histories when a numerical-capacity record is missing or invalid, while skipping gravity-transfer and drop mass construction.

The optional `--shipping-capacity-qualification FILE` input uses the strict `numi.cardboard.shipping-capacity-qualification.v2` key/value schema in `matter/tools/cardboard_shipping_capacity.hpp`. It is generated from six admitted native runs by:

```sh
python3 matter/tools/cardboard_stage_resolution_study.py capacity-qualification \
  --reference <shared-middle-run> \
  --mesh-coarse <coarse-mesh-run> --mesh-middle <shared-middle-run> --mesh-fine <fine-mesh-run> \
  --time-coarse <coarse-dt-run> --time-middle <shared-middle-run> --time-fine <fine-dt-run> \
  --output-properties <new-output>/shipping-capacity.properties
```

The shared middle run must be the same directory in both triplets. Every level must contain complete accepted solver certificates, a complete nominal/half-speed sensitivity pass, and the full 70% inside-height compression endpoint. Each triplet must have two admitted adjacent comparisons; all three compared metrics (peak, full-curve RMS, and work) must decrease from coarse-to-middle to middle-to-fine. An unchanged metric remains inconclusive without an independently measured noise floor. The finest pair must also meet 2% relative peak, 5% normalized force-curve RMS, and 2% work-difference limits. The sidecar binds the exact source physics/program fingerprints, source-material binding digest, assembled Matter and rigid-sidecar checkpoint hashes, nominal curve SHA, peak sample and endpoint, and both hashed triplet reports. The probe independently compares those identities with its current run and checks report-file hashes before permitting any gravity or drop mass branch. CPU failure-injection checks cover changed curve/checkpoint identities, descriptive-only mesh status, incomplete timestep levels, increasing or unchanged adjacent differences, above-tolerance curve difference, and a peak-only prefix.

This gate qualifies use of the numerical Fpeak for load sizing only. It does not qualify the shipping material grade, identify local starch/HMA fracture parameters, or establish physical-box validation. The existing `triplet` command remains descriptive; only the stricter `capacity-qualification` command writes the admission sidecar. No sidecar is supplied by default, so ordinary compression runs retain complete histories but do not instantiate heavy masses.

## Impactor footprint convention

The probe measures upward-facing exterior triangles from the fixture-free settled accepted FEM checkpoint, stores their projected X/Z bounds in `shipping-outer-roof-measurement.json`, and binds the measurement to the checkpoint SHA plus source and compiled-world fingerprints. The full impactor plan is 1.2 times those measured extents; centering and quarter-width offset use the same bounds. A missing/invalid roof measurement blocks gravity/drop scenario construction without discarding completed compression records. This geometry-derived sizing is still not impact contact qualification.

## Native source44 bounded checks

The integrated source44 build succeeded with the reusable tool families and explicit roof-bound loading geometry. Its bounded moving-BOX capacity-overflow and swept-path rejection cases073/074 pass, including accepted persistent-state rollback. Case075, the separated accepted-translation instrument, fails strict nonlinear acceptance; tool relocation is therefore implemented but not yet numerically qualified. The compiler and footprint contracts do not establish a shipping-scale physical transaction.
