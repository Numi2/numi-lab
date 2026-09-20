# Human source joint equalities: NHEQ2

NHEQ2 adds the 51 source scalar-joint equality rows to the existing coupled
Human/Matter solve. It preserves the pinned MuJoCo 3.12.0 classic acceleration
and positive-compliance semantics. It does not replace them with a hard
kinematic projection. The canonical Human remains 157 bodies, 129 configuration
coordinates, 128 velocity coordinates and 416 source muscles.

**Evidence status:** source row/algebra checks, isolated Metal row/preconditioner
probes, the constrained native prepared root and both Brain integration cases
pass on Apple M4 Pro with Metal API validation. The authored v4 Brain case
preserves exact rejected-root retry and joint publication. Each Brain case
covers eight accepted roots at 100 microseconds, only **0.8 ms** of physical
time. Sustained behavior and complete anatomical/material qualification remain
open. The retained artifact receipt distinguishes these evidence layers.

## Source and ABI ownership

The Human exporter requires MuJoCo **3.12.0**, source Euler integration and
classic inverse-weight semantics; `diagexact` is rejected. NHEQ2 keeps the
quartic coordinate relation, reference coordinates, `solref`, `solimp` and
source `dof_invweight0` for both dependent and master joints. A fixed-master row
has invalid master indices and zero master inverse weight. These are source
parameters, not fitted native stabilization constants.

The file has an 80-byte header, file ABI 2, policy 1
(`NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC`), explicit `REFSAFE` flag and
112-byte rows. The header binds the source SHA-256 shared with NHRIGID and the
complete source row count. The GPU row ABI is independently versioned at 1.
The canonical payload SHA-256 is
`12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477`.
The source-prefix check preserves the existing NHRIGID and NHEQ1 bytes exactly.

NumanX configuration v4 requires an authored Matter world plus the immutable
NHEQ2 payload. Its layout is 192 bytes: header, nested v3 at offset 8, equality
path at 176 and expected nonzero equality FNV-1a64 at 184. Nested v3/v2 headers
remain explicit. The v3 Human fingerprint still admits the base
NHRIGID/NHMYO/NHCNT world; the resulting runtime model identity additionally
mixes `FNV1a64("NHEQ2")` and the exact equality payload fingerprint. Changing
constraints therefore changes the live physical source identity.

Native construction validates payload identity, source/dimension agreement,
policy, flags, finite scalar rows, positive source inverse weights, scalar
coordinate indices and unique dependent joints. Dependent-master chains require
a separately qualified source profile and are rejected. Missing or invalid
NHEQ2 cannot silently produce an unconstrained v3 world. Swift selects
`mrnx_bridge_v1_runtime_create_v4` only when its equality descriptor is present;
old v1/v2/v3 callers retain their explicit existing behavior.

## Coupled physical law

All row coefficients are frozen once at the accepted `q0/v0`, as required by
the selected source acceleration policy. For each row:

```text
phi = q_dependent - q_dependent_reference - polynomial(q_master - q_master_reference)
J0  = [1, -polynomial_derivative]       (only the dependent term for a fixed master)
a_ref = -damping * (J0 v0) - stiffness * impedance(phi) * phi
R = max(1e-15, (1 - impedance) / impedance * (source_invweight_dependent + source_invweight_master))
b_delta = h * a_ref - J0 * (v_free - v0)
lambda = R^-1 * (J0 * delta_v - b_delta)
residual = other_coupled_impulse - A0 * delta_v - J0^T * lambda
```

The kernels retain source impedance clamping/interpolation, positive and direct
`solref` formats and the source `REFSAFE` time-constant clamp. Here `R` uses the
source classic inverse-weight approximation; it is not a newly computed exact
`J M^-1 J^T` diagonal. Positive `R` permits exact algebraic elimination of the
row multipliers. The equality contribution to the Newton action is
`J0^T R^-1 J0`; the unknown velocity layout remains full dimensional.

The existing source predictor remains authoritative:
`v_free = v0 + h A0^-1 (source_force - source_bias)`, with frozen effective
tangent `A0 = M(q0) + armature + h D`. Matter's `delta_v` is relative to this
predictor. The exact candidate and committed Human integration use that same
velocity. Source positions and velocities are never snapped onto the quartic
relation after solving. In particular, the canonical source pose's maximum
initial equality defect is **0.05241920054** in its source coordinate units;
that defect is retained and enters the compliant source force law.

Matter builds a separate private preconditioner
`B = A0 + J0^T R^-1 J0`. Positive Cholesky rank updates start from the read-only
source factor `L0`, preserving small source pivots; no projected inverse or
singular full-space projector is substituted. The B factor and inverse action
are numerical solver machinery. **The physical reaction delivered to Human
remains `A0 delta_v / h`, exactly once.** Replacing it with `B delta_v / h`
would introduce a second equality contribution. The original A0 factor,
accepted `q/v`, source muscle authority and caller-owned command timeline are
unchanged by constructing B.

Owning code is `src/metal/NumanXRuntimeV1.mm` for source admission,
`matter/include/numi/matter/human_equality_gpu.h` for the GPU ABI,
`matter/src/metal/human_equality.metalinc` for preparation/residual/action/factor
and inverse kernels, and `matter/src/runtime.mm` plus
`src/metal/MetalNumanXHumanMatter.mm` for same-root borrowed orchestration.
`apps/numanx_human_equality_probe.mm` owns the isolated arithmetic probe.

## Evidence and remaining qualification

The CPU oracle uses canonical packed FP32 row inputs promoted into MuJoCo
FP64. It compares four 51-row cases without stepping a trajectory. Its retained
`numi.human.nheq2-fp64-oracle.v1` result reports:

| Source/algebra quantity | Maximum error |
| --- | ---: |
| Position / Jacobian | `2.22e-16` / `4.44e-16` absolute |
| Reference acceleration / regularizer | `4.55e-13` / `1.73e-18` absolute |
| Eliminated versus full KKT velocity / dual | `4.81e-14` / `7.80e-14` absolute |
| Central-difference operator | `2.99e-12` relative |

This establishes the selected row law and Schur algebra for those inputs. The
KKT comparison uses an independent synthetic SPD A0; it is not a full native
Human dynamics or complete MuJoCo trajectory comparison. The source result and
prefix identity are retained in Human's
`Docs/media/numanx-source-equalities-20260908/` (raw licensed inputs remain local).

The macmini canonical Metal probe covers 51 rows, two environments and three
cases with a synthetic SPD source factor. With Metal API validation enabled,
`equality-preconditioner-canonical.log` reports:

| Metal versus FP64 quantity | Error |
| --- | ---: |
| Preparation / residual / action | `2.28e-7` / `4.77e-7` / `9.99e-8` |
| Residual finite difference | `4.75e-4` relative |
| B factor / inverse action | `1.55e-7` / `1.72e-6` relative |
| Inverse residual | `4.65e-5` |

It also verifies byte-identical replay, immutable source L0, untouched unrelated
vector channels and rejection of negative/nonfinite pivots. The four-row
synthetic probe separately exercises impedance/direct-reference cases. These
are isolated GPU arithmetic checks, not coupled outcome or performance proof.

The full-body probe reports eight rejected equality-admission cases and the
expected 51-row mixed source identity `9ae01c156c71d2a9`; twelve authored-world
negative cases also pass. Its constrained prepared physical root passes the
existing convergence tolerance with three FGMRES iterations. The source-aware
B inverse replaces the legacy A0 inverse for preconditioning; the adapter
requires exactly one encoded inverse authority, plus the existing mass action,
exact candidate and publication callbacks.

Both Swift integration cases pass with Metal API validation (2 tests, 0
failures, 37.869 seconds): the source-constrained authored world and the explicit
legacy fixture. They cover Brain proposal/application, atomic joint publication,
rejected-root restoration, exact retry and terminal timeout handling. No
publication or convergence tolerance was relaxed.

Nine native CTests pass with Metal API validation, covering owner transaction,
v4, exact candidate, attachments, legacy/authored full body, the equality kernel
and two Matter physics cases. The strict adapter allocator regression passes
on plain Metal, including actual completed-command-buffer address reuse. Its
validation-layer run cannot reproduce the required address reuse within the
fixture's 64 attempts; that prerequisite failure is retained, and its assertion
was not weakened. It is not counted as a passing validation-layer test.

Validation also exposed previously invalid Metal API use. The owning Human
operator now reserves aligned 16-byte threadgroup scratch; the adapter copies
its bounded status record from private storage into a shared diagnostic buffer
on the same physical command buffer. The forced-reject proposal binds a full
128-byte inert witness instead of a 96-byte owner-status record. The force-reject
shader never reads that witness and cannot accept from it. Swift test snapshots
use explicit checked readback ranges; production physics and sensor buffers
remain private.

Retained development failures include the original convergence failure,
preconditioner callback-accounting failure, private-buffer read errors and
undersized forced-reject witness. Later passing logs supersede those failures
for the named boundaries; the original records remain available.

The source and artifact receipt is retained in Human's
`Docs/media/numanx-source-equalities-20260908/`. It binds the native/Brain
revisions, binaries, metallibs, immutable inputs and exact launch configuration.
Standing for 60 seconds across 20 seeds, recovery across 100 frozen impulses,
walking at all three registered speeds, anatomy/material validation and
complete-stack performance remain unrun. A passing bounded transaction does
not close those gates.
