# Source 118 Metal depfile overlay

This package proposes a CMake-only dependency-tracking correction for the
source 118 production `NumiMatter.metallib` and the cardboard paper diagnostic
metallib. It does not change shader contents, material equations, runtime
behavior, or any qualification result.

`cardboard_paper_check.metal` includes `../src/matter_kernels.metal`, which
recursively includes the production FEM/contact/FGMRES Metal source and shared
headers. The source 118 custom commands list a partial include closure as
`DEPENDS`; the paper command's `.metalinc` glob does not include imported
`.h` files. Reproduced `xcrun metal -MMD -MF ... -MT ... -E` depfiles identify
transitive inputs absent from those explicit lists: 13 for the paper check and
8 for `NumiMatter` (including `gpu_types.h`, `constraint_ir_shared.h`, and
`accepted_state_apply_gpu.h`). A header-only edit to such a file therefore
does not reliably rebuild the corresponding metallib.

The patch adds a compiler-generated depfile to each Metal compile and attaches
it with CMake's `DEPFILE` property. It retains the existing explicit
`DEPENDS` entries as direct prerequisites. The repository requires CMake 3.28,
so the custom-command depfile support is available under the project's minimum
version.

Validation performed:

- `xcrun -sdk macosx metal -help` advertises `-MMD`, `-MF`, and `-MT`.
- Both source 118 Metal inputs were preprocessed with the exact include paths
  and compiler mode, producing readable depfiles with 39 paper and 38
  production inputs. These are compiler input inventories, not runtime tests.
- `run_depfile_cmake_test.py` builds a minimal Metal kernel with a transitive
  `.metalinc` under both Ninja and Unix Makefiles, edits only that included
  leaf, then confirms CMake reruns Metal compilation through the generated
  depfile.
- `git apply --check overlay.patch` succeeds against the frozen source 118
  snapshot's `matter/CMakeLists.txt`.

The helper-host review found no concrete input-bound defect in source 117's
`cardboard_source370_material_point_check.mm`. It checks exact hashes and the
package/tet/material identities, reads the candidate node buffer with an
exact byte-count check, and calls production FEM deformation/material-frame
helpers. The accepted-state values converted to FP32 match the emitted
accepted-state words. Native run 382 completed as a point-only diagnostic;
both 16- and 32-update local projections remained invalid with terminal
failure 6, which the checker records rather than promoting to a pass. This is
not a full-step or coupon result.

## Bindings

- Frozen source: `source-scaled-material-merit-118`
- Frozen-source manifest SHA-256:
  `90a55444fdb2991d520a95a82b2711b4eea1e2e58023e374fba446cc1a933dda`
- Source 118 `matter/CMakeLists.txt` SHA-256:
  `26be9449b499be7c1e5d4ac4120e5e89bda7a8d7c3afcd4f655fe5d7c8ff2052`
- Source 118 `cardboard_paper_check.metal` SHA-256:
  `4660f78dc594adcdcdd4cab6d6943114d16f54f5d1a3cc327270718e16a5e181`
- Source 117 `cardboard_source370_material_point_check.mm` SHA-256:
  `f02bae340982ad889f5e48488d82c42d25394c82b5af96bc4a51338277c25f21`
- Patch SHA-256: see `validation.json`.
