# Independent FP64 free-BOX inertia Jacobian check

This is a standalone CPU mathematics check of the angular inertia chain used by the free rigid BOX operator. It does not call Metal or the production checker and is not native qualification.

The oracle models the exact authored fixture inputs after canonicalizing GPU-resident source fields to FP32 and promoting those values to FP64. It evaluates

`q = Exp(dt * (omega0 + DeltaOmega)) * q0`,

`R = -I_world(q) * DeltaOmega`, and `A = -dR`, including the spatial left-Jacobian map `dtheta = dt * J_left(dt * (omega0 + DeltaOmega)) * dDeltaOmega` and the derivative `dI = [dtheta]x I - I[dtheta]x`.

Run:

```sh
python3 oracle.py
```

The six basis and three mixed directions were checked with central differences at `epsilon=1e-6`; the worst relative error is recorded in `report.json` and guarded below `1e-8`. The `1e-7` level also remains accurate in FP64. The FP32-native checker’s smaller-step refinement remains inconclusive due to candidate quantization, which this double-precision check deliberately avoids.

Native case 219 used the uncorrected checker SHA `4e0d3f1f…aa230`; the corrected sign/branch checker SHA is `855b816c…dbb11f`. Thus 219 does not validate that correction. A fresh native run bound to the corrected checker is required. The old native case and its receipt are preserved.
