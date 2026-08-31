# NumiLab Human exact knee articular-contact preflight

NumiLab now has a reusable, nonvisual small-deformation contact operator for
the seven cartilage/meniscus surface pairs authored in the bilateral
`NHKNEE1` Open Knee(s) oks003 payloads. It is intentionally separate from the
payload's twelve ligament/tendon collision pairs.

The operator builds exact closest-point correspondences from the named source
triangles, gives every slave node a tributary surface area, and scatters each
compressive traction to the three master nodes with barycentric weights. The
result is frictionless and tensile-free. Each contact sample therefore adds
equal-and-opposite force, and its master scatter preserves the application
point used for the moment check.

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

## Evidence boundary and next integration gate

This is `preflight`, not `one_step` or `sustained` Human evidence. The current
operator does not yet resolve penetration inside the owning articulated
Human/Matter transaction, does not scatter accepted contact reactions into
the femur/tibia/patella generalized force, and uses fixed reference
correspondences appropriate only for small deformation. The right side is a
mirror of oks003, not an independently segmented right specimen.

Promotion requires live current-surface contact search, unilateral
nonpenetration, reaction coupling without duplicate force ownership, rejection
rollback, replay, and a sustained flexion/compression trajectory with bounded
contact pressure, area, cartilage strain, meniscus strain, and energy.
