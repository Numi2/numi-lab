# Registered layer facet ancestry regression

This synthetic, data-independent test checks exact facet-to-RegisteredSurface row/stable-ID mapping, optional legacy behavior, filtering after material reordering, and the boundary-area consumer. It does not consume a private candidate file.

Trust boundary: the current explicit parent-volume CSR validator checks finite positive fractions, sorted/material-compatible parent contributors, per-child fraction sum, and aggregate parent-volume closure. When CSR is present it does not recompute geometric child/parent overlap. The test reports volume_csr_geometric_overlap=not_checked; source geometry construction or an independent overlap audit must establish that property.

Portable CMake invocation from the repository root:

    cmake -S matter -B build -DBUILD_TESTING=ON
    cmake --build build --target numi-matter-registered-layer-facet-ancestry-check
    ctest --test-dir build --output-on-failure -R '^matter\.compiler\.registered_layer_facet_ancestry$'
