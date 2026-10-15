# Selector-5 layout-check erratum

The sealed `graph-schwarz-native-063` package's host check validates deterministic patch construction, CSR coverage, overflow rejection, and storage offsets. Its source182-named case is a synthetic count-and-size fixture, not the captured source182 topology.

In `overlay/matter/tools/regional_graph_schwarz_layout_check.cpp`, the case sets `source182Nodes = 846`, marks every node free, excludes nodes 10, 30, 50, 70, and 90 from a synthetic sliding-window tetrahedron chain, and then checks for eight 256-node patches plus five singleton patches. It does not load the actual source182 mesh with 882 total nodes, its 36-fixed-node mask, or compare the canonical graph-membership digest `9d9625...`.

Therefore the README's “source182 layout check” and the fixture's “Reproduce source182's measured graph shape” wording overstate what this test establishes. The matching patch counts and byte sizes are useful allocation regressions only; they do not prove actual source182 patch membership or native selector-5 factor provenance. The sealed package is intentionally left unchanged. A follow-up capture must bind the actual node map and fixed mask and compare their canonical membership digest before any source182-layout claim.

The source182 matrix/pivot count arithmetic remains internally consistent for the asserted synthetic shape: 13 patches, 2053 patch-node incidences, 6159 scalar DOFs/pivots, and 4,718,637 dense matrix floats. This erratum does not invalidate the generic layout/overflow checks or Metal compilation; it narrows the interpretation of the named fixture.
