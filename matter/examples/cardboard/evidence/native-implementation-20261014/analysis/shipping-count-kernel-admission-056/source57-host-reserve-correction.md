# Source57: corrected shipping preflight host reserve

This is a CPU-only correction to the source56 allocation preflight. The source56 JSON is preserved byte-for-byte and is now explicitly superseded for reporting zero host source/compile bytes in its exact-topology scenarios.

The new preflight applies the same coarse host-side planning factors as `estimateShippingPrebuildMemoryBudget()` in `matter/tools/cardboard_shipping_memory.hpp`:

- source mesh: `256 * nodes + 512 * tetrahedra`;
- compiled world: `128 * nodes + 256 * tetrahedra`;
- generated Metal source reserve: 64 MiB;
- upload staging reserve: equal to compiled-world bytes.

For the authored shipping topology (528,444 nodes, 1,477,728 tetrahedra), this gives 891,878,400 B for source mesh, 445,939,200 B for compiled world, 67,108,864 B for generated Metal source, and 445,939,200 B for staging. Source + compiled + generated totals 1,404,926,464 B. These are explicit planning assumptions matching the probe's prebuild formula; they are not measured vector capacities, allocator overhead, generated-source/compiler peaks, or observed host/process memory. The 16 GiB whole-process admission remains incomplete.

The combined modeled allocation adds the Metal envelope, host source/compile inventory, and retained checkpoints. Upload staging and one readback reserve occur in both component subtotals but are counted once in the combined model.

For stride-15 sparse K=65,536, no-free-rigid, scratch-alias-on scenario, the corrected modeled allocation plus tangent cache is 12,909,476,422 B (12.023 GiB). The same case without scratch aliasing is 13,253,285,030 B (12.343 GiB). The modeled totals are below 16 GiB under the coarse reserve, but both remain rejected by current shipping admission/count/arena limits. This does not establish that a real full-shipping process fits in 16 GiB. The default-capacity compact case models 27.019 GiB with tangent cache; the bounded 131,072-slot compact scenario models 13.301 GiB but still fails other admission limits. Selector-2 and selector-4 regional preconditioner estimates remain above the current 256 MiB cap.

Validation performed on CPU only:

- `cmake --build build-cardboard --target numi-matter-cardboard-shipping-memory-check -j2`
- `./build-cardboard/numi-matter-cardboard-shipping-memory-check` — passed
- `python3 -m py_compile matter/tools/cardboard_shipping_allocation_preflight.py` — passed
- regenerated the source-bound preflight report; no Metal runtime or GPU was initialized.

The report is `shipping-allocation-preflight-corrected-057.json`, SHA-256 `73c4f51628ed1f819f4fd259e7845832011933e45cb55a7c011528fa30bfc6f5`. It binds the updated preflight C++ source (`173c163f8d0d964556db4bf4b11c28dca6f20639a9191279efdc1bb553640bd2`), Python runner (`62d1bc77aef9b90298f4245b7ecfa3f9e00f69207353ea341ffdc4966b72e921`), and CPU check (`58b385fe0afa0566bd33c7354dd4ac1f684fa8dcc283fe8c05e7583525de02db`).

The source56 report remains unmodified at SHA-256 `4a328687615caa53ecf0bd8d21798a46df609d4a9ea3e2ec9a97cd6ac420dbfd`; its sparse no-free-rigid exact-topology host mesh/compile values were zero. The older source55 estimate remains retained as an underestimated allocation model, not qualification.
