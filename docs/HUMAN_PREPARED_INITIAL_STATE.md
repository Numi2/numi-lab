# Prepared Human initial conditions

NumanX construction v7 extends the complete v6 authored-world/source-physics
contract with an immutable NHINIT payload and its nonzero FNV-1a fingerprint.
NHINIT1 and NHINIT2 remain byte-identical. NHINIT3 adds an exact, construction-
only handoff of prepared support multipliers to Matter's accepted history
owner. Existing v1–v6 construction retains the source default state. There is
no live reset API or host stepping/control path.

The canonical 96-byte little-endian envelope is:

| Offset | Field |
|---:|---|
| 0 | eight bytes `NHINIT1\0` |
| 8 | u32 ABI 1 |
| 12 | u32 header size 96 |
| 16, 20, 24 | u32 nq, nv, source muscle count |
| 28 | u32 scalar size 4 |
| 32, 36 | u32 flags/reserved, both zero |
| 40 | u64 composed Human source fingerprint before adding NHINIT1 |
| 48 | u64 exact authored Matter world fingerprint |
| 56 | u64 timestep in microseconds |
| 64 | 32-byte source archive SHA-256 |
| 96 | FP32 q, then v, then four FP32 values per source muscle |

NHINIT2 retains all NHINIT1 offsets, uses a 160-byte header, carries the exact
nanosecond clock at offset 96, and optionally carries the compensated root
translation at offsets 104–151. Its reserved bytes at 152–159 remain zero.

NHINIT3 retains all NHINIT2 offsets and uses this canonical extension:

| Offset | Field |
|---:|---|
| 0 | eight bytes `NHINIT3\0` |
| 8, 12 | u32 ABI 3, u32 header size 224 |
| 32 | flags: bit 0 compensated root, bit 1 prepared support history; bit 1 required |
| 160 | SHA-256 of the complete raw NHCNT byte image, 32 bytes |
| 192 | u64 raw NHCNT byte count |
| 200 | u32 NHCNT payload ABI, 1 or 2 |
| 204, 208 | u32 source record count, u32 expanded runtime row count |
| 212, 216 | u32 history record bytes 16, u32 encoding 1 |
| 220 | u32 reserved zero |
| 224 | existing q, v and muscle records, then one history float4 per expanded support row |

Encoding 1 stores `{tangentImpulseWorldX, tangentImpulseWorldY,
tangentImpulseWorldZ, normalImpulse}` in N·s. The tangent vector must be
orthogonal to the bound ground normal, the normal impulse must be nonnegative,
and the resultant must remain inside that row's Coulomb cone. NHCNT1 uses file
order. NHCNT2 uses primitive order; capsules expand endpoint A then endpoint B.
The exact NHINIT3 extent is
`224 + 4*nq + 4*nv + 16*muscleCount + 16*expandedRowCount`.

The muscle values are excitation, activation, fibre length in metres and fibre
velocity in metres/second. Excitation/activation lie in `[0,1]`, fibre length is
strictly positive and every scalar is finite. State dimensions must match the
loaded source. The runtime checks root quaternion normalization and source,
world and clock identity. It rejects trailing bytes and incomplete state;
decoding failure preserves the destination. The legacy decode entry point
rejects NHINIT3. Its bound overload requires an expected raw NHCNT SHA-256,
byte count, ABI and source/expanded counts, so a history cannot be rebound by
matching only its row count or the rigid-source digest.

Prepared q/v remain separate from `EngineModel.defaultQ/defaultV`. Source rest
coordinates and source anatomy remain unchanged. Mass ownership establishes the
final body frames before the prepared state is applied. Authored FEM attachment
position and velocity must agree with this same state at the existing FP32
packing tolerance. The runtime never relocates nodes to make a package pass.
The prepared source target includes NHEQ2, NHLIM1 and optional tissue ownership;
the NHINIT domain and exact payload fingerprint then enter Human program and
publication identity. All later physical state remains the resident owner's.

`RuntimeConfiguration::humanSupportInitialHistories` is the only support seed
handoff. Empty retains legacy all-zero initialization; otherwise its exact
environment-major extent is `environmentCount * supportContactCount`. Matter
uploads those values into `humanSupportHistoriesAccepted`; its existing
checkpoint, candidate, commit, rollback, snapshot and proof paths retain
ownership thereafter. The seed is never written into the contact descriptor's
reserved `.w` word and is never reapplied as a spring or force.

`encodeNumiHumanInitialState` is the native offline serialization interface.
The qualification probe also accepts:

```
metalrobo_numanx_fullbody_bridge_probe --prepared-stance-fixture \
  CERTIFICATE_LOG OUTPUT_DIRECTORY NHCNT_PAYLOAD NHEQ2_PAYLOAD NHLIM1_PAYLOAD \
  [TIMESTEP_MICROSECONDS [NEWTON_ITERATIONS \
  [FGMRES_RESTART FGMRES_ITERATIONS RELATIVE_RESIDUAL]]]
```

Exact-nanosecond authoring keeps certificate and imported-state provenance
separate:

```
metalrobo_numanx_fullbody_bridge_probe --prepared-stance-fixture-ns \
  CERTIFICATE_LOG OUTPUT_DIRECTORY NHCNT_PAYLOAD NHEQ2_PAYLOAD NHLIM1_PAYLOAD \
  TIMESTEP_NANOSECONDS [NEWTON_ITERATIONS \
  [FGMRES_RESTART FGMRES_ITERATIONS RELATIVE_RESIDUAL]]

metalrobo_numanx_fullbody_bridge_probe --prepared-state-fixture-ns \
  PREPARED_NHINIT OUTPUT_DIRECTORY NHCNT_PAYLOAD NHEQ2_PAYLOAD NHLIM1_PAYLOAD \
  TIMESTEP_NANOSECONDS [NEWTON_ITERATIONS \
  [FGMRES_RESTART FGMRES_ITERATIONS RELATIVE_RESIDUAL]]
```

The stance form authors a new NHINIT from the certificate and rounds each
support `force * exact timestep` directly at nanosecond precision. The state
form only rebinds an already composed NHINIT whose source identity matches the
current rigid, muscle, support, equality, and limit payloads; source drift
fails closed instead of relabelling stale state.

The legacy Newton-only override remains accepted. If any FGMRES field is
supplied, restart width, total budget, and relative residual must all be
present. The receipt records the requested value plus the executable FP32
residual and its exact bits, so a stricter diagnostic world cannot be mistaken
for the default fixture or silently promoted into calibrated physical
evidence. The same nested solver override is accepted by the microsecond
fixture modes.

Fixture authoring requires a fresh output path. The pack, NHINIT, and receipt
are written into a sibling staging directory and published together with an
exclusive atomic rename; an existing destination is never merged, truncated,
or replaced. The leaf must be new and its parent directory must already exist
so the parent entry can be durably ordered. The published directory remains
owner-private (`0700`). A failed pre-publication attempt retains its uniquely
named staging directory for inspection instead of recursively deleting through
a pathname. The FGMRES total budget is operationally capped at 1024 iterations.

This helper reads the saved native compiler q/muscle records and per-row normal
support forces, binds the exact supplied NHCNT1 or NHCNT2 bytes, rounds each
`force * exactTimestep` once to FP32 impulse, serializes NHINIT3, and compiles
the existing three tiny pelvis samples at that pose. Its receipt publishes the
raw support SHA-256, byte count, ABI and source/expanded counts. Both accepted
certificate forms retain those same five identity fields; authoring requires
their exact equality with the supplied NHCNT before assigning any force row.
The source-compliant JSON remains schema v1 with these additive provenance
fields.

## Sustained exact-runtime evidence

With the five exact inputs exported as `MRNX_EXACT_WORLD`,
`MRNX_EXACT_INITIAL_STATE`, `MRNX_EXACT_SUPPORT_CONTACT`,
`MRNX_EXACT_JOINT_EQUALITIES`, and `MRNX_EXACT_JOINT_LIMITS`, run:

```
metalrobo_numanx_exact_runtime_horizon_probe \
  --roots ROOTS --timestep-ns 12500 --uniform-excitation EXCITATION \
  --snapshot-step SELECTED_ROOT --snapshot-dir ABSOLUTE_NEW_PATH
```

The snapshot path must be a new leaf under an existing parent. The harness
requires exactly root 1 and `SELECTED_ROOT`, validates their semantic identities
and canonical filenames, and adds a no-replace run manifest binding argv,
device/runtime/world/clock identities, five input hashes, configure-time Git
revision, and snapshot file/payload hashes. Its success line deliberately says
`sustained-execution-only audit_required=true full_behavior=false`.

Audit both snapshots separately with
`tools/audit_numi_human_production_owner_snapshot.py` and a digest retained
outside the artifact via `--expected-payload-sha256`. A horizon completion and
snapshot audit establish repeatable runtime and numerical evidence only; they
do not establish standing, physical, biological, performance, or production
qualification.
It is an explicit qualification fixture, not anatomical tissue registration.
It does not apply NHEQ1 reaction forces to the source-compliant solver.

Curved support keeps capsule endpoints as separate physical rows. A device
reduction groups contiguous rows by immutable source geometry into the existing
ten touch receptors. It conserves total normal/tangential impulse and normal
centre of pressure, reports the minimum gap and impulse-weighted slip velocity,
and uses the nearest witness when unloaded. Bad rows invalidate their receptor.
All supplemental channels and the optional neuron-culture consumer use that
derived view. Matter retains sole authority over physical contact rows and
acceptance; sensing never writes them.

Admission does not certify equilibrium, tissue calibration or standing/walking.
Loaded source-compliant contact/equality/limit/fibre response, anatomical tissue
and sustained behavior require separate accepted-trajectory evidence.
