# Sparse private-contact prebuild wiring overlay (source 075)

This is an external, source-bound overlay against frozen snapshot `source-completed-graph-tail-and-signed-contact-oracle-075` (parent revision `35bfb518c542a306f563cc364ef8496837068b9a`). It changes only `cardboard_shipping_memory.hpp`, `cardboard_probe.mm`, and `cardboard_shipping_memory_check.cpp`. It does not alter runtime, Metal kernels, physics, contact ordering, acceptance, or CMake, and no GPU experiment was run.

## Change

The shipping prebuild path now parses the same `NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY` and `NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_PARITY` values as the compiled-world admission path, then carries `requested`, `capacity`, parity, six-DOF rows, and the prospective bounded free-BOX scope into `ShippingPrebuildArenaOptions` and `estimateShippingMemoryBudget`.

The prospective scope is recognized only for the native fixed shipping scene recipe. Its JSON field is explicitly called `bounded_free_box_scope_candidate`; the compiled-world proxy/body predicate and runtime encoder remain mandatory. Sparse samples do not compact canonical pair/history storage.

The prebuild pair domain counts every prepared FEM node against every rigid proxy. For the exact reference shipping blank it uses the known prepared count of 528,444 nodes after the optional starch-side split; otherwise it uses the larger of nominal and 25%-padded nodes. Tetrahedra use the exact reference count for the reference source and padded counts for other geometries. The shipped default model remains a dense rejection at about 30.55 GB.

With the reference recipe and explicit `K=65,536`, the prebuild rigid sparse arena is admitted at 36,113,232 modeled bytes using a 6,341,328-pair upper bound. The independently derived exact topology has 6,335,568 contact-eligible node/proxy pairs, so the upper bound is larger. Overall required residual descent remains rejected by the existing FEM full-contact node/tet/arena caps; this overlay does not make the shipping physics runnable.

## Verification

- `cardboard_shipping_memory_check.cpp` compiled as C++20 with `-O2 -Wall -Wextra -Wpedantic -Werror` and passed.
- `cardboard_probe.mm` passed frozen-075 Objective-C++ `-fsyntax-only` with the build’s definitions, include paths, and warning flags.
- `prebuild_probe.cpp` emitted `prebuild-contract-075.json` with default dense rejection, sparse rigid admission, and the still-rejected overall full-contact arena.
- `patch -p1 --dry-run` succeeded against frozen 075.

`source-overlay.json` records base and overlay file hashes; `patch-075.diff` is the exact integration patch. This is a prebuild-admission correction, not runtime allocation evidence or solver/physical qualification.
