Boundary node mapping preparation, 2026-10-24

The app now uses explicit object-local node rows for registered outer, inner-skin,
and deep-fat boundary presentation. The existing paired owner initializes exactly
the old mapping. Additional interior volume nodes can be represented without
pretending they have registered surface identities; the optional initial CSV uses
layer=3 and row=local-volume-node for those nodes.

Validation: portable C++ boundary-map regression passes with strict warnings;
full native app compilation/link pass. Source-only initialization emits byte-identical
7233 FEM node positions/masses, detailed skin and 86 body poses against apps008.
No physical step, new geometry, anatomy acceptance or throughput result is claimed.
The general unpaired volume preparation owner and geometry repairs remain pending.
