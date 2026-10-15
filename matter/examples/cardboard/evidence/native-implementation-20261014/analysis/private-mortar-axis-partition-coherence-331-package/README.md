# Projected partition trajectory coherence and full normal closure

This package is a narrow supplement to the exact 330 axis-normal whole-parent candidate. It validates that shared projected vertices in the bounded (2–4 cell) left and right mortar partitions follow the same endpoint trajectory. Cells that share a start vertex but diverge at the finish are rejected, preventing a crack in the projected partition from being hidden by matching each patch ID independently.

The CPU suite passes 49 checks. One new geometric control moves a fully covered interface from +0.125 normal gap to -0.125, crossing at t=0.5; the complete parent and same-ID projected partition certificate remains valid throughout. This demonstrates that the certificate does not rely on a positive terminal gap. The separate negative control rejects a shared start vertex that tears apart by the endpoint.

The contact shortcut is reached only after active, non-dormant mixed-mode cohesive patches exactly cover both parents. For the admitted axis-normal path, the relative gap is uniform across each horizontal parent and affine in time. The active mixed cohesive law evaluates a tensile branch for positive normal gap and a compressive `stiffness.z * gap` branch for nonpositive normal gap. Thus the cohesive law owns closure throughout an opening, contact, and crossing interval; the ordinary collision query is not needed for an uncovered complement because the projected cover is complete and stable. This is a bounded source-level argument, not a native force/contact qualification. It assumes that this active mixed cohesive law is the intended compression response for the fully covered interface.

The full Metal source compiled and linked with Metal 4.0, `-O3`, and `-fno-fast-math`; two unrelated FGMRES maybe-uninitialized warnings remain. No GPU/native run was made. The candidate remains unqualified for arbitrary poses, tangential motion, deformation, inactive/dormant patches, incomplete coverage, or other contact laws.

Apply this patch after `axis-normal-relative.patch` from package 330:

```sh
git apply /absolute/path/to/axis-partition-coherence.patch
```

See `manifest.json` for exact base, patch, source, and build bindings.
