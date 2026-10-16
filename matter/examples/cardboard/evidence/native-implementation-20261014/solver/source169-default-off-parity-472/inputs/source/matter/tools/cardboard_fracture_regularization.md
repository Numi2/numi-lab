# Tetrahedron crack-plane geometry reference utility

`cardboard_fracture_regularization.hpp` computes a selected crack-plane's
intersection with one tetrahedron in its reference configuration. Given
reference volume `V_e` and the actual polygon area `A_e` cut from that same
tetrahedron, it returns the characteristic length

```text
l_e = V_e / A_e
```

For a crack-band law whose dissipated work density is `Gc / l_e`, the
element-local energy budget is

```text
(Gc / l_e) V_e = Gc A_e.
```

The implementation intersects the plane with all six tetrahedron edges,
deduplicates cut points, orders the resulting triangle or quadrilateral, and
computes its reference area. It uses the absolute tetrahedral determinant, so
reversing input vertex winding does not change the volume. A material-frame
normal is transformed by a validated proper rotation into reference space;
the plane offset is a signed reference length from the element centroid along
that transformed unit normal. A reference-frame normal is already expressed in
the mesh reference axes.

This is a geometry primitive, not a damage model. It does not choose a crack
plane, supply `Gc`, initiate or evolve damage, resolve a damage tangent, or
prove mesh objectivity of a constitutive implementation. For a plane with no
positive-area intersection, the result explicitly has zero area and no
characteristic length. A crack plane exactly coincident with a shared
tetrahedron face is reported per element; a mesh-level assembler must define
one-sided ownership or avoid that case to prevent counting the same shared
surface twice. Areas here are element-local reference cut areas, not the
separate reference areas of authored cohesive faces.

The CPU test uses analytic unit-tetrahedron cuts, a rotated material frame,
reversed tetrahedron winding, uniform scaling, an exact centroid subdivision,
and `Gc*A` conservation. The numerical `Gc` values in the test are synthetic
unit-test inputs only. No value is asserted to be a measured or calibrated
paperboard fracture energy. The test can be compiled independently while a
project target is wired:

```sh
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  matter/tools/cardboard_fracture_regularization_check.cpp \
  -o /tmp/cardboard_fracture_regularization_check
/tmp/cardboard_fracture_regularization_check
```

Before use in a native constitutive update, the caller must bind a source-
supported initiation/evolution law and fracture energy, decide how element
crack normals are selected and transported, and verify accepted/candidate
state and energy consistency on a refined mesh.
