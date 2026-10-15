# Shipping-size runtime admission and memory gaps

## Source103 continuation

Native353 now passes all16 bounded mortar CCD cases, including the four valid
motions rejected by source101. This clears that geometric regression only.
Combined-contact356 passes its synthetic transaction checks; free-body355
still fails at its first-fatal Newton7, captured without changing production
history in358/359. No shipping arena admission, measured process-memory peak,
accepted shipping step or fixture-free checkpoint follows. The explicit-flute
counts and independent runtime limits below remain. See
[the source103 solver report](solver-milestone-source103.md).

## Reference011 and experimental097 continuation

Selected all-direction material comparisons now resolve the FP64 and native
material gates, with exact endpoint arithmetic errors retained. They do not
remove shipping admission limits. Experimental CCD318 passes12 of16 synthetic
cases and still rejects four multi-patch sweep motions. No accepted shipping
FEM step, complete native arena measurement or fixture-free checkpoint exists.
See the current solver ledger for precise scopes and preserved failures.

## Source095 continuation

The selected private-trial capture now validates129 arrays/19,467,248 bytes,
with complete capture-on/off production parity. Its physical fixture still
rejects. Private GPU314 retains all14 case inputs and statuses:9 pass and5
expected-accept mortar motions reject. Four patches are representable, but
partition/complement and shared-boundary motion remain unqualified. The
shipping CPU exact-rational audit proves zero overlap for635 recorded sides
across497 remaining sliver pairs; it is not a Metal runtime repair.

The explicit-flute shipping counts, independent arena caps and modeled12.42GB
envelope below remain unchanged. No complete native allocation/physical-step
admission or fixture-free shipping state exists. Source095 material/capture
milestones do not remove those limits. Detailed current results are in the
solver ledger; historical source091/083 statements below retain their scope.


Source091 repairs the scoped paper FB value arithmetic; native300 passes all
retained point projections, independently of shipping admission. Source090's
three-patch parent-pair union now permits the formerly rejected activation299,
but strict closure and private shared-boundary CCD301 still fail. The current
shipping HMA topology also contains parent pairs with four mortar patches,
beyond that bounded union. Passing CPU builds or point checks does not remove
these contact, mesh-count, arena or full-process allocation requirements.

The latest bounded solver work is documented in
[solver progress](solver-progress-20261014.md). Shared FEM/rigid Armijo fractions
are repaired and selected graph assembly is measured, but strict nonlinear
contact and complete coupon convergence remain open. The experimental
selector5 preconditioner is limited to contact-free coupons and cannot admit
the shipping box. The modeled sub16GiB configuration remains an allocation
plan, not a measured complete native arena or whole-process peak. Shipping
assembly, released checkpoint, capacity-derived masses, gravity crush and
impact therefore remain unexecuted.

Source083 updates memory accounting for the opt-in ordered cohesive page path.
It counts a96-byte force page per face/environment, capped at8MiB, and all
persistent interface data; it does not assume that all260,835 faces fit a
legacy cloned trial arena. Invalid page selectors fail closed. The independent
8,192-node,32,768-tet and contact-capacity restrictions remain, and native280
does not complete even the small legacy/tiled comparison. This accounting
change is not measured shipping allocation or execution admission.

Run287 subsequently retains byte-identical legacy/tiled failed prefixes. The
one-face fixture rejects closure; the three-face fixture rejects activation.
Source087's full-arena readback admits the small free-body fixture's reserved
18,336 contact slots under the independent 32 MiB diagnostic cap. Its first
manifest has a retained nonfinite-JSON serialization failure. Source088 fixes
serialization only;291 validates the strict manifest and all140 array hashes,
and291/292 preserve physical state and Armijo histories with capture on/off.
The failed three-face fixture also exposes a single-parent-pair lookup that
cannot represent multiple active mortar patches. Its union/complement must be
supported in contact and private CCD; a compiler rejection alone is not support.
Neither bounded readback nor failed-prefix parity removes shipping limits.

The heavy-object footprint convention is now explicit:20% larger means
1.2 times the roof area, preserving its aspect ratio (linear scale sqrt1.2).
The previous1.2 scale on each axis implied44% more area. No accepted shipping
impact used that former convention, and historical evidence is preserved.

The source076 capability inventory checks the actual prepared mesh: 528,444
nodes, 1,477,728 tetrahedra, 260,835 cohesive faces, 1,054,876 exterior faces
and 6,335,568 canonical rigid pairs. At one environment, the examined index
products fit U32; this does not clear the scope or arena gates. Sparse private
rigid storage still scans the full pair domain, production contact remains
dense, and the self-contact node-incidence builders scan every active contact
for every node. The latter is an independent N-by-K work cost. Cohesive aliasing
does not bypass the current pre-alias 8 MiB trial-byte admission check.

The separate prebuild wiring repair carries the explicit sparse selector,
capacity and bounded free-BOX scope candidate into the same checked sizing
helper. It admits only the private rigid-storage subgate (about 36.1 MB at
65,536 candidates); full shipping FEM/contact/cohesive execution still rejects.
Neither this estimate nor the approximately 12.42 GB modeled envelope is a
measured simultaneous allocation or a physical transaction. Read-only hardware
checks found 16 GiB on numi and 24 GiB on macmini, so neither is a larger-memory
continuation for this local 24 GiB machine.

Date: 2026-10-15
Scope: audit of the native shipping runtime and its sizing helpers, plus the admission-consistency repair and source-bound Fpeak numerical-capacity gate. No production solver, contact law, tolerance, mesh, or certificate behavior was changed. No GPU transaction was run for this audit.

## Current continuation boundary

The current duplicated-starch/closure mesh is 528,444 nodes, 1,477,728 tetrahedra and 260,835 cohesive faces; the source014 topology below is retained as a historical count anchor. Source057 allocation models can fall below 16 GiB with explicit scratch aliasing and bounded contact reserves, but whole-process simultaneous peak and shipping-scale native admission remain unmeasured. The complete shipping box has executed no physical step.

Sources060–063 repair selected-Newton sidecar capture, canonical FP32 diagnostic inputs, latched activation geometry restore, rotated world-inertia checking and the shipping adapter's final-contact-buffer phase. These component repairs do not remove the large-mesh/private-arena admission limits. The latest complete coupon attempt still rejects after three of 27 controls. The native graph-Schwarz conditioning path and nonlinear rotating-body path are separate work in progress.

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

The compiler creates an authored rigid-contact pair for every contact-eligible continuum node and every rigid proxy (`matter/src/compiler_impl.cpp:3183-3211`). The prebuild estimator deliberately counts every FEM node against every proxy (`matter/tools/cardboard_shipping_memory.hpp:724-730`). The archived source014 shipping mesh has 453,378 total nodes and 858,908 one-owner exterior tetrahedron faces. Counting the unique vertices on those exterior faces gives 428,940 contact-domain nodes. These mesh counts are bound to `/Users/home/cardboard-implementation-20261014/archive/shipping-preflight-cpu-20261014-014/mesh.json.gz`, SHA-256 `de3388be8a356d6e9a54c3e9ee789a4ba136bd745ff4fe512dd7b02998e01449`; they are an anchor for that archived mesh, not a claim about later remeshing. The corresponding dense pair upper bounds are:

| Proxy count | All 453,378 FEM nodes | 428,940 compiled contact-domain nodes | Contact-domain pairs / 65,536 cap |
|---:|---:|---:|---:|
| 2 | 906,756 | 857,880 | 13.1× |
| **12 current shipping proxies** | 5,440,536 | **5,147,280** | **78.5×** |
| 16 | 7,254,048 | 6,863,040 | 104.7× |
| 39 historical shipping reserve | 17,681,742 | 16,728,660 | 255.3× |

The current shipping scene uses twelve proxies: three reusable scoring families, each with one punch and two rectangular rails, plus compression platen, floor and finite loading body. Every crease still requires its own physical indentation, withdrawal and release; family tools relocate through native accepted translation commands. The authored scene fits the 16-proxy scope; source48 prebuild accounting still used 39 proxies/5 bodies, corrected by source49 shared 12-proxy/9-body constants. For the source014 mesh, the exact all-contact-domain product is 5,147,280 rows (78.5× the 65,536 pair cap). The 5,440,536 total-node product is a separate looser upper estimate, not the compiled surface-domain count. Even the two-proxy contact-domain case exceeds the cap by 13.1×. The historical 39-proxy allocation calculation remains for comparison. At the source014 twelve-proxy count, the prebuild estimate's per-pair shared cost `sizeof(NMContactPairGPU)+16` bytes is about 157 MiB; the runtime private trial sample alone is 160 bytes per pair, about 785 MiB, before active maps/incidence. Runtime limits the trial samples to 36 MiB and 65,536 pairs. A smaller deformable-contact slot count cannot solve this independent rigid-pair scaling problem.

## CPU swept-candidate prototype (not runtime-integrated)

`matter/tools/cardboard_contact_candidate_domain.hpp` implements a bounded CPU broadphase prototype. It accepts a conservative sphere around each contact-domain node's complete trial path, swept bounds for fixed-orientation translating capsules and oriented boxes under the native linear-tool-translation contract, and clearance parameters. The FEM sphere can conservatively enclose nonlinear solver paths; the endpoint is separately supplied and must lie within that sphere, while the full path bound remains the caller's certified input. A moving rigid path without the authored linear-translation contract is rejected. The helper uses `max(supportDistance, minimumGapFloor) + candidateMargin` plus coordinate-scaled FP32 roundoff. Capsule bounds cover both axis endpoints and radius over the translated sweep; box bounds use its fixed orientation and translated endpoint centers. The node's whole-path sphere is applied around the accepted position, so a path that crosses a tool while both endpoints are clear is not discarded. Any missing/nonfinite FEM path bound, unsupported shape, rotating tool, inconsistent capsule translation, duplicate key, or capacity overflow rejects the whole result. Pair keys are sorted stable `(objectIndex,nodeIndex,proxyIndex)` tuples. Overflow returns the exact required candidate count and an empty list; the helper never truncates.

The CPU test `matter/tools/cardboard_contact_candidate_domain_check.cpp` compares emitted candidates with independent double-precision capsule segment-distance and segment-to-oriented-box narrowphase oracles. It covers a moving-node crossing with clear endpoints, moving capsule and BOX crossings with clear tool endpoints, a far no-hit node, a nonlinear FEM path enclosed by its accepted-position sphere, the minimum-gap floor, ordering stability, unbounded motion, unsupported rotation, inconsistent capsule endpoints and exact overflow reporting. It passes as a broadphase contract test only. It is not a native all-pairs/operator/history equivalence test and is not wired into the runtime. The helper still enumerates the bounded node/proxy input combinations on CPU; a production GPU spatial hierarchy, buffer-lifetime proof, stable accepted-history remap, same-transaction rollback, and parity with the current all-pairs contact residual remain required before replacing the compiled pair table.

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

For orientation only, adding the default-capacity private deformable arena to the case032 runtime-resident measurement gives 17,151,771,152 bytes, about 15.97 GiB, before completing the real rigid/tool and cohesive allocations. This is an extrapolation, not a measured simultaneous peak. The current twelve-proxy shipping prebuild implies rigid pair/sample costs that fail the existing caps. At 160 bytes per pair, its private rigid samples would require 823,564,800 bytes using the source014 contact-domain node count, or 870,485,760 bytes under the looser all-FEM-node bound, before incidence, dense stable histories or FEM/self-contact arenas. The shipping prebuild uses `T/5` as a cohesive-face reserve when a cohesive law is requested (`matter/tools/cardboard_probe.mm:4257-4260`); that is a planning heuristic, not a proven upper bound for the compiled flute interfaces. Admission must use the exact compiled face count before runtime initialization and reject if the trial arena does not fit.

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

## Exact source52 shipping allocation estimate

The source-bound CPU preflight `shipping-allocation-preflight-3231b850.json` includes the proper duplicated starch interfaces and all closure/tool domains. The declared topology has528,444 nodes,1,477,728 tetrahedra,260,835 cohesive faces (97,984 starch plus162,851 closure),1,054,876 exterior faces, and6,335,568 canonical rigid pairs. These replace the source014 no-starch-domain estimates above for the intended world; the earlier measurements remain bound to their original topology.

With an explicitly bounded131,072 contact-slot arena, compact immutable exterior faces and15-state materials, the model estimates8.10 GB before the optional484.7 MB tangent cache, or8.59 GB with it. The default contact reserve and noncompact exterior domain estimate23.20 GB. These are allocation models, not measured simultaneous peaks or runtime admission. A pair of checkpoint host copies adds approximately3.83 GB of modeled storage overlap and must be included in the peak measurement.

Source52 permits the canonical rigid pair index domain above65,536 only when checked sparse-private admission succeeds. A65,536 candidate reserve plus the canonical pair slot map requires36,090,192 bytes (34.42 MiB), within the unchanged36 MiB reserve without the dense parity reference. It still excludes free rigid unknowns. The full world remains rejected by the8,192-node/32,768-tet/4,096-cohesive-face bounds,281.5 MB private self-contact scratch against128 MiB,125.2 MB private cohesive scratch against8 MiB, and regional coarse arenas of456.6/837.1 MB against256 MiB. Conservative sparse candidate sizing does not qualify full shipping contact or impact.

The source52 probe now enables substrate self-contact with the starch-interface model. Intact-interface exclusions handle tied coverage; independent closure contact remains available after separation. No shipping transaction or fixture-free assembled checkpoint has been accepted under this configuration. One GPU worker is retained throughout.

Source079 adds opt-in deterministic contact incidence and parallel graph LU
search. Native260/261 qualify only prepared lists/factors; production263/264
first-control parity passes. Bounded graph fallback admits independent coupon
layouts within existing arenas, but actual nx16 case265 rejects. The full
shipping mesh still lacks native FEM/contact/cohesive admission and measured
peak allocation. No limits are raised by these changes.
