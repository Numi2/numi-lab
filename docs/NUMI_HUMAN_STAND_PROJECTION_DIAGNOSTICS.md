# Numi Human stand projection diagnostics

`MRNumiHumanStandStatusGPU` retains the terminal velocity residuals of the
bounded `NumiHumanStand` coupling path both immediately before and immediately
after its final exact equality-coordinate projection.  The diagnostic is
observation-only: it does not change a force, impulse, iteration budget,
regularization, or state update.

For contact and source scalar limits, each residual is evaluated with the
same pre-step point Jacobian, contact gap, and source coordinate position that
the coupled sweep used; only the terminal candidate velocity changes.  It is
therefore a local linearized comparison, not a collision re-query after
integration.  The equality residual uses the integrated coordinate immediately
before or after its exact projection.  Each `float4` stores:

1. maximum normal-contact target-velocity residual;
2. maximum source position-limit target-velocity residual;
3. maximum equality target-velocity residual; and
4. the maximum of the preceding three values.

The persistent Human trace exports both vectors per accepted sample alongside
q/v, tendon residuals, virtual work, constraint impulses, and the existing
equality-projection overwrite diagnostics.

## Bounded M4 Pro result

With Metal API validation, the one-step frictionless minimal triad at 12.5 us
and 64 coupled sweeps contained one normal contact, one equality, and one
active upper limit.  Its pre/post residuals (m/s or rad/s as applicable) were:

| Row family | Before final equality projection | After final equality projection |
| --- | ---: | ---: |
| Normal contact | `3.8849307770760788e-08` | `0` |
| Source upper limit | `0` | `2.2428295665122278e-07` |
| Equality | `2.2428295665122278e-07` | `0` |

The inactive-limit control retained a `7.2759576141834259e-12` contact
residual before and after projection, with zero limit and equality residuals.
The no-equality control recorded zero diagnostic values by construction.
Two repeated probe runs produced byte-identical diagnostic output.

This isolates a concrete interaction: in this bounded triad, the exact final
equality overwrite clears its own residual while moving the active upper-limit
row away from its target.  It does not establish the cause of the full-body
common-duration refinement failure, validate a replacement formulation, or
qualify passive standing, anatomy, physiology, walking, or safety.  In
particular, `NumiHumanStand` remains a separate bounded diagnostic path from
the source-compliant NHEQ2/NHLIM1 Human/Matter runtime.

## Reproduction

Build and run the focused owner probe on an Apple Metal host:

```sh
cmake --build <build-dir> --target metalrobo_numi_human_stand_coupling_probe
MTL_DEBUG_LAYER=1 <build-dir>/bin/metalrobo_numi_human_stand_coupling_probe
ctest --test-dir <build-dir> --output-on-failure \
  -R '^(numi_human_source_route_precision_cpu|numi_human_source_route_precision_metal|numi_human\.stand_coupling_minimal)$'
```

The next discriminator is to use the same two vectors in the full persistent
trace and compare the affected rows, impulses, tendon residuals, mechanical
work, q/v trajectory, and common-duration refinement against the existing
CPU/FP64 references before changing the coupled formulation.
