# Shipping regional preconditioner and full-contact execution audit

Date: 2026-10-15. Read-only source audit plus checked arithmetic; no GPU run or production edit is represented here.

## Geometry and measured-evidence boundary

The source-bound CPU preflight `shipping-allocation-preflight-source054.json` records 528,444 prepared nodes, 1,477,728 tetrahedra, and 260,835 potential cohesive faces (97,984 starch interfaces plus 162,851 HMA closure faces). It binds the base geometry to 453,378 nodes and the same tetrahedron count. This is an authored shipping-sized mesh and allocation model, not a native shipping solver run. The separate 10,467,551,760-byte resident measurement is source032, a smaller base FEM world without the present shipping tooling or cohesive topology; it must not be described as the measured peak for this mesh.

## Current regional RAP arena

`matter/src/runtime.mm:726-786` constructs the source basis for one explicit FEM object. Each compact constituent region contributes up to six translation/rotation trial fields. Selector 2 uses `6 * regionCount`; selectors 3/4 reserve the conservative maximum of 32 modes before rank reduction and add three float4 rows per node for the block-fine correction. With three roles, selector 2 has 18 modes; the selector-4 basis can contain up to 26 useful fields (18 regional plus eight bending candidates), but runtime allocation/admission conservatively sizes it at 32. The 256 MiB admission cap is checked before these buffers are allocated.

For `N=528,444`, `E=1`, and float4 size 16 bytes, each dense mode/node matrix costs `16*N*M` bytes:

| Mode count | One P, effective-P, or AP matrix | Planner regional arena | Selector-4 block fine rows | Planner total |
|---:|---:|---:|---:|---:|
| 18 | 152,191,872 B (145.14 MiB) | 456,581,968 B (435.43 MiB) | 0 | 456,581,968 B |
| 26 | 219,832,704 B (209.65 MiB) | 659,510,608 B (628.96 MiB) | 25,365,312 B (24.19 MiB) | 684,875,920 B (653.15 MiB) |
| 32 | 270,563,328 B (258.03 MiB) | 811,708,432 B (774.11 MiB) | 25,365,312 B (24.19 MiB) | 837,073,744 B (798.30 MiB) |

The planner conservatively represents coarse scalar/pivot arrays as float4-sized items. Runtime buffers use float4 for source P/effective-P/AP and block rows, but actual coarse matrix, pivot, RHS, coefficient, and validity buffers are smaller scalar allocations (`matter/src/runtime.mm:5102-5157`). Even the runtime’s selector-2 planning estimate is 435 MiB, and selector 4’s 798 MiB estimate is far over the 256 MiB limit. Thus shipping is rejected before regional allocation; shrinking the GPU’s global working-set target does not remove this explicit arena gate.

AP is a persistent preconditioner input, not just RAP construction scratch. The precondition application reads every mode’s P and AP at each node to form

`z = P c + D^-1 (r - AP c)`, with `c = (P^T AP)^-1 P^T r` (`matter/src/metal/fgmres.metalinc:1746-1788`; block-fine selector 3/4 uses the same AP at 1791-1837). It is also streamed through `nm_fgmres_regional_build_rap` to form every generally nonsymmetric coarse entry `P_i^T A P_j` (`fgmres.metalinc:1564-1593`). At 18/26/32 modes this RAP reduction performs up to 171,215,856 / 357,228,144 / 541,126,656 nodal dot terms, respectively. Building AP applies the matrix-free production operator once per basis column: 26,599,104 / 38,420,928 / 47,287,296 tetrahedron actions, respectively.

Tiling AP only while forming RAP does not solve the persistent-AP requirement in the application kernel. A memory-saving but unqualified alternative is to store no AP matrix: retain effective P, form `P c`, apply the exact production A to that vector for each preconditioner application, and use `D^-1(r-A(Pc)) + Pc`. RAP can still be assembled by one AP column at a time into the small coarse matrix. This removes roughly 210 MiB at 26 modes or 258 MiB at 32 modes, but adds one full FEM/contact/cohesive A action per FGMRES preconditioner application. The standard FGMRES Arnoldi A(z) remains separate. It changes FP32 summation order, so qualification must compare the streamed A(Pc) to a dense-AP-vector action across representative vectors and accepted histories, including contact/cohesive terms, with no symmetry assumption and no relaxation of physical residual/certificate tolerances.

That alternative alone still exceeds the 256 MiB regional cap: P + effective-P + AP-free nodal scratch at 26 modes is about 447 MiB before the block-fine rows. Reaching the current cap would require an additional proven reduction, such as storing only one nodal basis matrix and regenerating/projecting it from immutable authored region data when grip masks change. Do not overwrite immutable P or skip mask updates without a transaction/lifetime proof. A source-validated compact basis representation is another option, but quantization would be a distinct conditioning-only change requiring native operator/preconditioner parity tests.

## Full-contact residual-descent admission is a separate blocker

The full-contact Armijo runtime path is bounded independently of the regional preconditioner:

- `matter/src/runtime.mm:8465-8479` requires one physics substep at microtick zero, rate exponent zero, at most four environments, at most 8,192 FEM nodes, one explicit FEM object, and at most 32,768 tetrahedra.
- Its dense box-contact private sample arena is limited to 36 MiB and the canonical pair domain to 65,536 pairs; it allows 2–16 authored proxies (`runtime.mm:8517-8563`, `detail::densePrivateContactPairDomainSupported` in `matter/include/numi/matter/sparse_private_contact_admission.hpp:12-19`).
- The full-contact private surface/BVH/sort/contact arena has a 128 MiB bound, node/tet/contact-slot count bounds, and at most 131,072 contact slots (`runtime.mm:4750-4880`, with transaction scope rechecked at 8688-8720). Cohesive private trials admit at most 4,096 faces and 8 MiB (`runtime.mm:8694-8699`).
- Sparse private contact compacts trial samples, but still addresses the canonical pair map, has a 36 MiB cap, and rejects free-rigid generalized unknowns (`sparse_private_contact_admission.hpp:14-17, 58-155`).
- Compiler selector 4 also excludes any generalized rigid rows, limits mesh/cohesive counts to the same 8,192/32,768/4,096 domain, and rejects dynamic/articulated proxies (`matter/src/validation_impl.cpp:580-624`). It can represent self-contact, mixed cohesive faces, and static/kinematic proxy contact within that bounded scope, but it does not cover the shipping finite-mass impactor’s six free rigid rows.

The preflight’s shipping mesh is 64.5x over the 8,192 node cap, 45.1x over the tetrahedron cap, around 96.8x over the 65,536 pair cap for the 6,335,568 proxy-node pairs, and up to 63.7x over the cohesive face cap. Even a deliberately bounded 131,072 contact-slot model estimates 281,522,368 B for FEM full-contact private arenas with compact surfaces and 1,064,375,428 B for rigid contact private storage; both are outside the respective 128 MiB/36 MiB private bounds, before accounting for source topology and snapshots. The current shipping starch recipe also disables deformable self-contact while the full-contact path requires it; that incompatibility must be resolved without removing required self-contact or silently changing the law.

## Minimal executable route without changing physical equations

1. Keep a small bounded native fixture that exercises the exact production operator with self-contact, active mixed cohesive faces, and kinematic tools; use it for selector-4 and private-contact parity. This can establish the operator contract but cannot qualify full shipping size.
2. For a scalable full-contact path, first reduce contact candidate/scratch size using a source-bound conservative broadphase/capacity argument or a streaming/contact-tile design. Preserve overflow rejection and rollback. The current 6.3-million canonical rigid-pair table and default `8*T = 11,821,824` deformable slots do not fit the existing private admission.
3. Add bounded multi-level coarse action using the unchanged A. The most direct memory experiment is streamed `A(Pc)` plus column-streamed RAP; measure the added full operator action per Arnoldi application and prove action parity against dense AP. Then address the remaining P/effective-P/block basis arena with a checked compact or regenerate-on-mask representation.
4. Extend solver admission in stages only after small native parity and rollback tests. Keep the 8,192-node/32,768-tet/65,536-pair/4,096-face scopes honest until a broader arena design is independently measured under the 16 GiB device budget. The full shipping solver and assembled physical box remain unqualified until those limits are addressed and full time/mesh convergence runs complete.

## Source anchors

- Shipping counts and arena estimates: `/Users/home/cardboard-implementation-20261014/shipping-allocation-preflight-source054.json` and `/Users/home/cardboard-implementation-20261014/shipping-allocation-preflight-3231b850.json` (older count model; keep the source054 528,444-node values distinct from older 453,378-node base).
- Regional mode basis/admission: `matter/src/runtime.mm:726-786`; allocation: `runtime.mm:5102-5157`.
- RAP build and application: `matter/src/metal/fgmres.metalinc:1564-1593,1746-1838`.
- Full-contact runtime allocation and gates: `matter/src/runtime.mm:4750-4880,8465-8720`.
- Compiler selector-4 scope: `matter/src/validation_impl.cpp:580-632`.
- Shipping memory planner: `matter/tools/cardboard_shipping_memory.hpp:390-420,465-553`.
