# Prepared Human initial conditions

NumanX construction v7 extends the complete v6 authored-world/source-physics
contract with an immutable `NHINIT1` payload and its nonzero FNV-1a fingerprint.
Existing v1–v6 construction retains the source default state. There is no live
reset API or host stepping/control path.

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

The muscle values are excitation, activation, fibre length in metres and fibre
velocity in metres/second. Excitation/activation lie in `[0,1]`, fibre length is
strictly positive and every scalar is finite. State dimensions must match the
loaded source. The runtime checks root quaternion normalization and source,
world and clock identity. It rejects trailing bytes and incomplete state;
decoding failure preserves the destination.

Prepared q/v remain separate from `EngineModel.defaultQ/defaultV`. Source rest
coordinates and source anatomy remain unchanged. Mass ownership establishes the
final body frames before the prepared state is applied. Authored FEM attachment
position and velocity must agree with this same state at the existing FP32
packing tolerance. The runtime never relocates nodes to make a package pass.
The prepared source target includes NHEQ2, NHLIM1 and optional tissue ownership;
the NHINIT1 domain and exact payload fingerprint then enter Human program and
publication identity. All later physical state remains the resident owner's.

`encodeNumiHumanInitialState` is the native offline serialization interface.
The qualification probe also accepts:

```
metalrobo_numanx_fullbody_bridge_probe --prepared-stance-fixture \
  CERTIFICATE_LOG OUTPUT_DIRECTORY NHCNT_PAYLOAD NHEQ2_PAYLOAD NHLIM1_PAYLOAD
```

This helper reads the saved native compiler q/muscle records, serializes their
FP32 state and compiles the existing three tiny pelvis samples at that pose.
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
