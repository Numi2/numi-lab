# Prescribed active fibre tension in the owning FEM transaction

Matter ABI 35 accepts an optional environment-major Float32 Cauchy-tension
buffer through `EncodeRequest::femActiveTensions`, with exactly one value per
cooked tetrahedron. Source-mesh arrays must be remapped to the compiler's
cooked tetrahedron order before a whole-mesh encode. The caller keeps that
borrowed buffer immutable until the command buffer completes. Matter evaluates
the stress and its geometric tangent inside the same FEM candidate/accepted-state
transaction. The reference fibre
comes from each element's selected material and, when authored, its immutable
source material frame. The full passive constitutive stress remains intact.

This path is for explicit **non-mixed** FEM. The existing mixed-FEM activation
path adds a pressure constraint that does not reproduce the Rodero cardiac
volumetric energy; a nonzero prescribed tension on a mixed element is rejected.
Every supplied tension must be finite, nonnegative and no greater than its
cell material's `maximumActiveTension`. Wrong count or undersized Metal buffers
fail on the host; invalid values on evaluated elements fail on the GPU and roll
back. Omitting the buffer preserves the previous path, and a supplied all-zero
buffer yields the same accepted FEM node state in the executable control.

`numi-matter-fem-active-tension-probe` compiles the source ventricular Guccione
passive law with a declared **synthetic 1050 kg/m3 density**, one framed
tetrahedron, three fixed nodes and one free node. On the local Apple M4, a
100 Pa tension accepted one step, shortened the free node by
`2.7474016e-7 m` versus zero input, and replayed bitwise. Over-limit, negative
and nonfinite tensions produced `NM_STATUS_NONFINITE_INPUT` and left the
accepted FEM node unchanged. A second run consumed Rodero case18 source cell 1,
its derived fibre frame and a source candidate tension of `89809.78125 Pa` at
100 ms. Its one-cell step accepted and replayed bitwise; the free-node movement
relative to zero input was `1.39437375e-7 m`. The source-cell test likewise
uses synthetic inertial density and three fixed nodes.

Build and run the focused owner tests:

```sh
cmake -S matter -B Build/cardiac-active-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build Build/cardiac-active-native --target \
  numi-matter-fem-active-tension-probe \
  numi-matter-cardiac-material-check \
  numi-matter-fem-material-frame-compiler-check
ctest --test-dir Build/cardiac-active-native -R \
  'matter.compiler.cardiac_material_source|matter.compiler.fem_material_frame|matter.runtime.fem_active_tension' \
  --output-on-failure
```

The full case18 wall has **zero accepted native electromechanical steps**. The
source activation reconstruction differs from the published CARP timing, and
case18 supplies no qualified stress-free reference, inertial density, physical
supports, valve/port interpretation or measured cardiac-motion validation.
This input is a source-bound mechanics ingress, not heartbeat qualification.
