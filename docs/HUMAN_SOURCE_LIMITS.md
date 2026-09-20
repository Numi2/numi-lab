# Source-compliant scalar joint limits

NHLIM1 adds the pinned MuJoCo 3.12 classic scalar-joint limit law to the
existing Matter Newton/FGMRES solve. NumanX configuration v6 requires the
authored-world v4/NHEQ2 owner plus an immutable NHLIM1 path and FNV-1a64
fingerprint. The three costal ownership fields are optional as one complete
group. Configuration v5 and older constructors retain their contracts.

The 80-byte little-endian envelope contains magic `NHLIM1\0\0`, ABI 1,
q/v dimensions, joint count, 80-byte record size, source count, classic policy
1, REFSAFE bit and two zero reserved words, followed by the source archive
SHA256. Each 80-byte record contains q/v/source-joint indices, a zero reserved
word, lower/upper range, margin, source `dof_invweight0`, solref[2], solimp[5]
and zero padding. Human's compiler requires every limited scalar source joint
to have exactly one native binding. The current full-body program has 122
joints, including six knee translations omitted by legacy hard-range flags
because their source reset lies just outside a tiny source range.

Construction validates immutable bytes, source identity, source clock,
native range agreement, unique coordinates/source IDs, finite normal FP32
parameters, inverse weights and policy. Source admission folds NHEQ2, then
NHLIM1, then optional NHTMASS1 into the Human fingerprint. NHLIM1 adds
`FNV1a64("NHLIM1")` and its payload fingerprint using the same two XOR/multiply
operations as NHEQ2. Base world admission still uses the original Human
fingerprint, before those domains. Program identity also binds limit rows,
dispatch and shader bytes.

At q0/v0, independently instantiate the lower and upper row when its signed
distance is strictly less than the authored margin. Each admitted side has
J=+1 or -1. Freeze source impedance I, stiffness K, damping B and
R=max(1e-15, (1-I)/I * dof_invweight0). With source timestep h:

```text
phi    = distance - margin
a_ref  = -B*J*v0 - K*I*phi
bDelta = h*a_ref - J*(v_free-v0)
lambda = min(0, (J*deltaV-bDelta)/R)    # negative physical impulse
force contribution to residual = -J^T*lambda
positive operator contribution = J^T*(1/R)*J when J*deltaV-bDelta < 0
```

The unilateral derivative is rebuilt at each Newton residual candidate and
reused unchanged for that candidate's Krylov solve. Both source-admitted
sides enter an SPD envelope preconditioner through positive rank updates of
the existing NHEQ2 factor. When a side releases this is an approximation to
the tangent; the actual operator continues to use the current active set.
No full stiff matrix is reconstructed or downloaded on the host.

The source law is memoryless. Its prepared rows and active tangent occupy
private per-runtime scratch, regenerated for each candidate/frame. Physical
q/v, muscle, contact and tissue state remain under the existing accepted-root
checkpoint/restore owner. The kernels never project q or v, commit a command
buffer, open a second physics queue or introduce a host stepping path.

Anatomy queries return admitted NHLIM1 ranges with the explicit
`MRNX_JOINT_COORDINATE_SOURCE_COMPLIANT_LIMIT_V1` flag. The reset coordinate
is retained exactly even when outside that compliant range. Clients must not
interpret these bounds as evidence of hard clamping or zero penetration.

`numanx.physics.human_source_limits` executes the checked-in independent
MuJoCo oracle. Human's `tools/qualify_joint_limit_source.py` regenerates it
using MuJoCo 3.12 `mj_forward` and `mj_constraintUpdate`, without stepping.
It covers both sides, strict margins, release, direct/positive solref,
REFSAFE on/off, variable/constant/clamped impedance, factor/preconditioner,
finite differences, replay, source-state preservation and failure isolation.
The extreme REFSAFE-off near-release case uses an explicit FP32 operand
cancellation bound; well-conditioned force checks retain 5e-4 scaled tolerance.
The prior failed absolute-result-only tolerance is retained in Human evidence.

This is source-law and bounded transaction qualification, not a material
calibration, sustained standing/walking result or complete anatomical loading
certificate. The legacy NHEQ1 visual horizon is a separate diagnostic path.

Source semantics: [MuJoCo 3.12 constraint engine](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_constraint.c).
