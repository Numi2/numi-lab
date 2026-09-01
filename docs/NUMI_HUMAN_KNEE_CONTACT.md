# NumiLab Human exact knee articular contact

NumiLab now has a reusable, nonvisual small-deformation contact operator for
the seven cartilage/meniscus surface pairs authored in the bilateral
`NHKNEE1` Open Knee(s) oks003 payloads. It is intentionally separate from the
payload's twelve ligament/tendon collision pairs.

The operator builds exact source-triangle correspondences from the named
surfaces and gives every slave node a tributary surface area. At runtime it
recomputes the closest point on that triangle in its current rigid pose. Face,
edge, and vertex regions therefore use the current closest-point direction,
oriented by the stored anatomical reference normal. The compressive traction
is scattered to the three current master vertices with the closest point's
barycentric weights. The result is frictionless and tensile-free; every sample
adds a collinear equal-and-opposite force pair and preserves world-origin
moment balance.

The pressure-overclosure law is

```text
k_i = (1 - nu_i) E_i / ((1 + nu_i) (1 - 2 nu_i) t_i)
k_eff = 1 / (1 / k_master + 1 / k_slave)
p = k_eff max(0, closure)
```

Cartilage uses the Open Knee/KneeHub elastic-foundation reference values
`E=12 MPa`, `nu=0.45`, and `t=3 mm`. Meniscus normal compression uses the
published radial/axial `E=20 MPa`, `nu=0.3`; thickness is measured from the
actual specimen mesh as `2 volume / (top contact area + bottom contact area)`.
The Open Knee(s) generation-2 models use deformable, nearly incompressible
cartilage and transversely isotropic menisci, so this efficient elastic
foundation remains a reduced v1 contact law rather than a replacement for a
fully coupled continuum model.

Primary source context:

- [Open Knee(s) specimen-specific model paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9832097/)
- [KneeHub/Open Knee documentation](https://simtk.org/docman/view.php/1061/11501/KneeHub_OKS.pdf)

## M4 Pro nonvisual qualification

The left and sagittally mirrored-right payloads each passed a 65-step
load/unload ramp across 69,701 exact contact samples and seven source pairs.
The 50 micrometre prescribed closure is a small-strain constitutive exercise;
it is applied to all registered pairs separately and is not a physiological
whole-knee load case.

Both sides passed:

- exact seven-pair admission and non-empty coverage per pair;
- compressive-only pressure and monotonic loading;
- force and moment balance near machine precision;
- `U = 1/2 F delta` elastic work/energy identity;
- bitwise peak replay; and
- exact restoration of the unloaded zero-force state.

Receipts:

- [`left-m4-pro.json`](media/numi-human-knee-contact-preflight-v1/left-m4-pro.json)
- [`right-mirrored-m4-pro.json`](media/numi-human-knee-contact-preflight-v1/right-mirrored-m4-pro.json)

## M4 Pro live Human one-step qualification

The live adapter cooks every one of the same 69,701 exact correspondences into
the Human transaction. Of those, 57,930 cross femur/tibia/patella rigid-body
ownership and generate balanced articulated wrenches. The remaining 11,771
are meniscus/cartilage interfaces owned by the same tibia body: they stay in
the immutable anatomy fingerprint and audit ledger, but cannot add a duplicate
rigid wrench until they have a relative deformable owner.

Both sides ran with the active four-muscle quadriceps/QAT/PTL chain, passive
ACL/PCL/MCL/LCL/QAT/PTL Matter FEM, source foot support, gravity, and joint
equalities in the same borrowed Apple Metal command buffer. The left accepted
1,077 closed samples, `0.000485220 m^2` active area, `0.0310359 N` normal
force, and `371.812 Pa` peak pressure. The mirrored right accepted 8,816
closed samples, `0.00387656 m^2`, `0.969647 N`, and `880.507 Pa`.

The left force/moment residuals were `7.68e-9 N` and `7.48e-9 N m`; the right
residuals were `5.96e-7 N` and `7.18e-7 N m`. Both retained positive FEM
deformation Jacobians, bitwise replay, rejected-step rollback, and the prior
approximately 2 kN extensor-chain force transfer.

The current-triangle owner was then checked on the left Human after the ABI 9
change. The accepted one-step transaction retained 600 closed samples,
`0.000267622 m^2` active area, `0.0146814 N` normal force, and `230.508 Pa`
peak pressure. Its force and moment residuals were `2.51e-9 N` and
`2.33e-10 N m`; FEM Jacobians remained `0.999760..1.000261`, replay was
bitwise, and rejected-step rollback passed. A deliberately rejected
plane-interior-only implementation admitted only 11 samples and is not
qualification evidence.

The same ABI 9 owner then passed a two-step `10 microrad` flexion trajectory.
The accepted ledger retained 25,293 to 30,987 closed samples and 4.632 to
16.431 N normal force. Trajectory maxima were 2,773.98 Pa pressure,
`1.59182e-6 J` stored energy, `1.02716e-4` layer-normal strain, and
`3.65661e-7 m` closure. Maximum force and moment residuals were
`2.16047e-5 N` and `2.46338e-6 N m`; deformation Jacobians stayed in
`0.999426..1.000620`, replay was bitwise, and rollback passed. This is a
bounded short trajectory, not sustained or physiological flexion. A requested
two-step 1 mrad run failed closed at the projected-anchor compatibility gate
before solving and is retained only as a diagnostic.

ABI 9 receipts:

- [`qualification.json`](media/numi-human-knee-current-triangle-v1/qualification.json)
- [`left two-step M4 Pro transcript`](media/numi-human-knee-current-triangle-v1/left-two-step-m4-pro.log)
- [`CPU regression`](media/numi-human-knee-current-triangle-v1/cpu-regression.log)
- [`Metal transaction fixture`](media/numi-human-knee-current-triangle-v1/metal-fixture.log)
- [`whole-body gate report`](media/numi-human-knee-current-triangle-v1/whole-body-gate.json)
- [front](media/numi-human-knee-current-triangle-v1/left-two-step-front.png),
  [oblique](media/numi-human-knee-current-triangle-v1/left-two-step-oblique.png),
  [side](media/numi-human-knee-current-triangle-v1/left-two-step-side.png), and
  [rear](media/numi-human-knee-current-triangle-v1/left-two-step-rear.png)

Receipts and full transcripts:

- [`left-m4-pro.json`](media/numi-human-knee-articular-live-v1/left-m4-pro.json)
- [`right-mirrored-m4-pro.json`](media/numi-human-knee-articular-live-v1/right-mirrored-m4-pro.json)

## M4 Pro accepted trajectory audit

ABI 8 adds an anatomy-derived maximum layer-normal compliance to every sample
and a fixed 4,096-step device audit ledger. The provisional contact record is
committed into that ledger only after the enclosing Human stand status accepts
the same step. Rejected and aborted transactions therefore cannot be counted as
trajectory evidence; replay deterministically overwrites the same accepted
step slot.

The synthetic M4 Pro fixture measured `0.9999 N`, `9999 Pa`, `0.009999`
maximum layer-normal strain, `0.0009999 m` closure, and `0.0004999 J` stored
energy. Its deliberately rejected second step rolled Matter back and left the
accepted history count at one; replay was bitwise.

The exact left knee then passed a bounded two-step run. Both steps retained
nonzero compression: the closed-sample range was `550..1077` and the normal
force range was `0.0101987..0.0310359 N`. Trajectory maxima were `371.812 Pa`,
`8.43778e-6` layer-normal strain, `4.90116e-8 m` closure, and `2.34765e-10 J`
stored energy. Maximum force and world-origin moment residuals were
`7.68e-9 N` and `7.48e-9 N m`; FEM deformation Jacobians stayed in
`0.999302..1.000699`, replay was bitwise, and rejected-step rollback passed.

Receipts:

- [`m4-pro-fixture.json`](media/numi-human-knee-articular-history-v1/m4-pro-fixture.json)
- [`left-two-step-m4-pro.json`](media/numi-human-knee-articular-history-v1/left-two-step-m4-pro.json)

## Evidence boundary and next integration gate

The prescribed 65-step CPU ramp remains `preflight`. Bilateral historical
coverage and the ABI 9 left current-triangle owner are qualified for one live
step; both the fixed-correspondence baseline and ABI 9 current-triangle owner
have bounded left two-step trajectories. Accepted contact wrenches enter
femur/tibia/patella generalized force in the owning Human transaction. ABI 9
recomputes the closest point and normal on the current paired triangle, but it
does not yet switch to adjacent facets or perform a global current-surface
search. Its explicit penalty is also not an implicit unilateral
nonpenetration solve. The right side is a mirror of oks003, not an
independently segmented right specimen.

The next promotion gate is adjacent-facet/global current-surface repair or
another bounded nonpenetration method, followed by sustained physiological
flexion/compression with pressure, area, cartilage/meniscus strain, energy,
replay, rollback, and failure criteria over time.

## Shared Human/Matter transaction infrastructure

The existing Numi Human tendon/FEM adapter now accepts optional internal FEM
contact samples. Contact does not own a second adapter, command queue, commit,
or external-force buffer. One Metal kernel evaluates closure from the current
accepted FEM nodes and accumulates slave/master forces through a validated
per-node incidence table after tendon traction assembly. Matter then solves
the combined load, and the existing fixed-node reaction kernel returns the
accepted cartilage/meniscus attachment reactions through the owning Human
body Jacobians.

The internal FEM-contact kernel remains an explicit fixed-reference penalty
on the previously accepted deformable state. The articulated articular kernel
is now different: it performs a current closest-point query on each paired
triangle and subtracts a `0.1 um` FP32 preload slop. Together they prove
same-transaction force transfer, not a global current-surface search, an
implicit contact solve, or unilateral nonpenetration.

A fresh Apple M4 Pro one-tetrahedron fixture passed with one mechanical sample
and one explicitly retained same-body sample:
the slave moved `19.8344 um` in the repulsive direction, combined anchor
reaction was `14.0519 N`, the combined NHTENDON/articular full-row result was
`-12.4389`, a malformed contribution table failed initialization, peak replay
was bitwise, and rejected-step rollback was verified. The machine-readable receipt is
[`internal-contact-adapter-m4-pro.json`](media/numi-human-knee-contact-preflight-v1/internal-contact-adapter-m4-pro.json).

This fixture proves transaction composition only. Bilateral anatomical
promotion comes from the separate 69,701-sample live receipts above.

## Reduced exact-surface articular wrench path

A full-resolution 12-region experiment admitted the exact six articular
volumes (194,729 total live nodes and 844,287 tetrahedra) and all 69,701
contact samples, but did not finish one replay-qualified step inside a
30-minute M4 Pro smoke bound. That path was rejected rather than promoted.

ABI 8 therefore provides a reduced path consistent with the existing
elastic-foundation law. Each cooked sample stores exact slave and closest
master points in their owning bone frames, the master-frame normal, tributary
area, reference separation, foundation stiffness, and the more compliant
layer's normal-strain-per-pressure coefficient. Metal evaluates closure,
reduces equal/opposite sample forces and moments to per-body wrenches, and
scatters those wrenches through the existing articulated-body Jacobians. It
uses the same borrowed command buffer, status, generalized-force arena,
replay, and rollback boundary; it does not create a second solver or CPU loop.

The M4 Pro two-body A/B fixture measured a `0.9999` generalized-force
correction while the Matter FEM state remained bitwise identical to the
no-mechanical-articular-contact run. An active sample mislabeled across the
same body failed initialization; a correctly typed same-body sample was
retained without generalized force. The audit measured `1.9998 N` body-force
L1, `9999 Pa` pressure, and zero force/moment residual. ABI 8 additionally
audits strain, closure, energy, and accepted trajectory history. Replay and
rollback passed. The current receipt is
[`m4-pro-fixture.json`](media/numi-human-knee-articular-history-v1/m4-pro-fixture.json).

The bilateral cook, force/moment balance, and one-step pressure/area gates pass,
and the left two-step history gate passes. Sustained loaded flexion remains
open. Meniscus relative motion and
fluid/poroelastic effects remain future continuum refinements rather than
claims of this reduced v1 law.
