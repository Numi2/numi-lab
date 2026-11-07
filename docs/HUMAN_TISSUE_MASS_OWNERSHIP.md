# Anatomical tissue mass ownership

`NumiHumanTissueMass` partitions a source body's discrete mass distribution
between the residual rigid body and final cooked FEM nodes. `NumiHumanTissueBinding`
lowers fourteen registered costal regions into one Matter world. Runtime
configuration v5 composes that world with the same rebased Human mechanics,
source joint equalities and existing joint transaction. This is an experimental
admission path; loaded thorax, experimental material validity and real-time
performance are not qualified.

## Mass and frame contract

The input measure is the final cooked `positionAndMass.w` at each registered
node, expressed in its donor's original COM frame. Source mesh volume or an
uncooked density integral must not replace it. Every node has one donor.

For original mass `M`, inertia `I`, tissue mass `m`, first moment `s` and raw
second moment `T`, the remaining mass is `R=M-m`, COM offset `c=-s/R`, and
central second moment `C=0.5*trace(I)*Id-I-T-R*c*c^T`. The residual inertia is
`trace(C)*Id-C`. Both `C` and inertia must be positive definite before and after
FP32 packing. This rejects impossible triangle inequalities as well as negative
mass. Packed zeroth/first/second moments must recombine within 16 float epsilons.
Axes remain fixed; the residual inertia is generally non-diagonal.

`rebaseNumiHumanTissueMassPartition` changes canonical body properties, both
sides of joint anchors and local shape positions atomically. A floating donor
root shifts both its reset position and translational velocity by `omega x c`.
It rejects a repeated/stale partition, a fixed donor root, and spatial
ConstraintIR that needs its own recook. The existing scalar NHEQ2 program is
independent of these local frame translations.

Every external source point must subtract the same offset. The costal compiler
recooks attachment locals and verifies that world-space cooked positions and
masses are bit-identical. Runtime v5 also rebases muscle sites/wraps, physical
point queries, support-contact locals, camera origin and source visual bounds.
The four basis queries per body refer to the new COM. Neither Swift nor Python
adds a physics stepping path.

## NHTBIND1

The offline Human compiler emits exactly 424 little-endian bytes:

| Offset | Record |
|---|---|
| 0 | Eight-byte `NHTBIND1` magic |
| 8 | Six uint32: ABI 1, regions 14, bodies, nodes, tetrahedra, reserved 0 |
| 32 | SHA-256 of actual NHCART1 bytes |
| 64 | SHA-256 of actual NHRIGID2 bytes |
| 96 | SHA-256 of the source registration JSON |
| 128 | Sixteen float64, row-major atlas-metres to world-metres transform |
| 256 | Fourteen triples of uint32 donor, sternal anchor, rib anchor body |

The transform must be one proper similarity with scale in [0.5,2]. Independent
endpoint fits, reflection and shear are rejected. The initial implementation
requires each region's three body indices to agree; articulated ribs require
an explicit volumetric donor map. Human verifies named bone mesh identities,
source model identity and source-body poses against the actual rigid bytes.
The registration is a source-bound candidate, not validated anatomical accuracy.
The current donor assumption is that costal volume belongs to the source gross
torso inertia; this assumption needs subject-level evidence.

## Runtime ABI and identity

`mrnx_runtime_config_v5` is 224 bytes: header at 0, nested v4 at 8,
NHCART1 path at 200, NHTBIND1 path at 208, expected binding FNV-1a64 at 216.
`mrnx_bridge_v1_runtime_create_v5` rejects an invalid nested configuration,
source/hash/coverage/topology mismatch, changed registered rest geometry,
incomplete or duplicate attachment ownership and impossible mass subtraction.
Existing authored-world admission then verifies the rebased attachment frames.
The unmodified v4 constructor rejects this rebased world against old COM frames.

The supplied Human fingerprint remains the base NHRIGID/NHMYO/NHCNT identity.
After NHEQ2 composition, runtime identity appends `NHTMASS1`, little-endian
binding FNV64 and world FNV64. This binds the changed physical mechanics to the
same prepared/proposed/applied/published root. The hot loop and rollback path
are the existing Human–Matter owners. v1 through v4 layouts remain unchanged.

## Reproduction and limits

Build `numi_human_tissue_mass_probe`, `numi_human_tissue_binding_check` and
`metalrobo_numanx_fullbody_bridge_probe` with the canonical full-body assets.
Run `ctest -R '^numi_human_tissue_mass_partition$'`. The binding checker takes:

```text
numi_human_tissue_binding_check BINDING CARTILAGE RIGID REGISTRATION \
  --metal --output-world=costal-rebased.nmatterpack
```

On an idle Apple Metal 4 device, set `MTL_DEBUG_LAYER=1`, `MRNX_COSTAL_WORLD`,
`MRNX_COSTAL_BINDING`, `MRNX_COSTAL_PAYLOAD`, and `MRNX_JOINT_EQUALITIES`, then
run `metalrobo_numanx_fullbody_bridge_probe --costal-tissue`. Its bounded
60-second hardware wait leaves the 10-microsecond physical step and solver
admission tolerances unchanged. It tests preparation and quarantine, not commit.
Use the NumiBrain authored-world joint-publication test with its v5 descriptor
for accepted publication and retry qualification.

The M4 Pro audit verifies eight full-body point/Jacobian cases and isolated
donor mass-matrix cases, with bit-identical replay. The 157-body dense Metal
mass-matrix diagnostic exceeds its supported bucket; whole-body dynamic mass
matrix qualification remains false. Costal contact is disabled explicitly,
rib/sternum anchors remain on torso 20, and material parameters are uncalibrated
population priors. See the Human repository's tissue-ownership evidence receipt
for exact input hashes, raw failures and measured outcomes.


### Experimental explicit donor and attachment binding (NHTBIND2)

The binding decoder also accepts a version 2 prototype. It retains the version 1
source hashes, registration, counts and fourteen region records, but permits
independent sternal and rib attachment body indices. Each region's donor field
must be the invalid-index sentinel; immediately following the 424-byte common
prefix are exactly `nodeCount` little-endian 32-bit body indices, one mass donor
for each cartilage node. The compiler uses these donors for mass partitioning
and the region attachment indices for anchor frames; the runtime validates each
cooked anchor against the appropriate endpoint rather than the mass donor.
Legacy version 1 bytes retain their original single-body semantics.

The format does **not** establish anatomical identity or an anatomical mass
partition. It cannot represent a node with contributions from several donor
bodies. The authoring process must establish provenance for each donor and
endpoint, verify the bodies and articulation actually exist and can move
relative to one another, and establish the mass partition before a patient or
atlas model may be treated as qualified. Distinct body indices alone do not
prove relative mobility. No such new anatomical mapping is supplied here.

The CPU binding regression uses synthetic identities and checks version 1
compatibility, separated version 2 identities and atomic rejection of malformed
bindings. A second synthetic fixture cooks a tetrahedron against a floating
two-body, one-revolute-joint model. Both donors are partitioned; sternal and
rib attachments reach the same ownership validator used during runtime
admission; their relative joint Jacobian differs by 0.01; and a corrupted
attachment is rejected. The equivalent fixed-root fixture is explicitly rejected
because rebasing a fixed-root donor requires an authored root-frame recompile.
Runtime asset shifts now use a candidate copy before publication, but this CPU
test does not construct a runtime or demonstrate rollback during a rollout.
Neither test qualifies Metal dynamics, loaded thorax response, biological
accuracy, or a surgical task. The existing synthetic
mass-partition invariants and earlier qualified Human evidence retain their
original scope.


#### Anatomical evidence still required

The fourteen cartilage regions come from the BodyParts3D costal-cartilage
payload; the existing source model maps its imported ribs and sternum to Human
torso body 20. A hash-bound atlas registration and this payload do not by
themselves identify separate anatomical rigid bodies. Before attempting an
anatomical NHTBIND2 authoring pass, retain:

* the exact atlas source, laterality, rib levels and mesh hashes for cartilage,
  sternum and each proposed rib, plus the source Human rigid/model revision;
* the registration transform and independently checked anatomical landmarks
  and endpoint-to-bone correspondence, including uncertainty;
* the construction and joint definitions for actual live sternum and rib
  bodies, and a test of relative point Jacobians rather than merely distinct
  body IDs;
* a volumetric, mass-conserving attribution from the original Human rigid mass
  to the cooked FEM nodes and residual bodies, with moments and positive
  residual inertia checked. Version 2 can assign only one donor per node;
  mixed donor contributions require an explicit extended schema;
* material provenance and loaded, held-out force/displacement measurements,
  including costal joint behavior. Synthetic attachment and kinematics tests
  do not provide those measurements.

A surgical continuation still requires the actual earned checkpoint and its
recorded identity; this binding fixture cannot substitute for that state.
