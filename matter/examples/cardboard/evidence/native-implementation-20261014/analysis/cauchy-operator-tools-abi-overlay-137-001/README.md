# Direct FEM operator checker ABI update for buffer 26

Source135's `nm_fem_apply_operator_elements` now has a required `device const float4* fallbackMetrics` at buffer index 26. These five direct operator checkers still bind indices 0–25. This overlay supplies an environment-count-sized zero `nm_float4` array at index 26 for each direct operator encoding. The buffer is zero for every environment, so it leaves ordinary operator modes (0, 1, and 2) inactive while making the new ABI explicit and safe. The coupled operator fixture expands its argument array and binding loop through index 26.

Changed files are limited to:

- `matter/tools/cardboard_coupled_free_box_operator_check.mm`
- `matter/tools/fem_regional_material_check.mm`
- `matter/tools/fem_material_frame_check.mm`
- `matter/tools/cardboard_gpu_check.mm`
- `matter/tools/fem_reference_check.mm`

No solver/runtime/kernel code, test policy, tolerance, or operator mode changed. The checks here are compile-only; no GPU was run and no scientific result is claimed. `verify_overlay.py` checks the source135 base hashes, confirms each direct call site binds a zero buffer sized to `environmentCount`, and verifies that the generated patch applies cleanly.
