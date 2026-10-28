# Physical skin/fat transfer checkpoint 209

The former detailed-surface ancestry does not define cooked tissue mass. Native preparation can now export the exact compiled parent cell ledger and separate node/material/source-donor contributions with `NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV`. This is load-time preparation; it adds no per-step CPU physics or duplicate state. The inherited mass-ownership changes distinguish the unique node/frame owner from split source donors and preserve rigid/FEM mass and moments.

Validated on SSH Numi (Apple M6): full native CMake/Metal build, 64 rigid-motion mass cases and 11 negative cases, whole-body mass/first/second moments, unpaired volume compiler tests with assertions enabled, and production C++ replay of 7,233 nodes / 28,632 cells / 9,644 source rows. Captured cell mass is 13.939679738169952 kg; cell/source/compiled nodal replay errors are zero. The donor has zero invalid Float32 cells, no nonmanifold faces and one connected skin/fat volume.

This checkpoint does not admit candidate033 as physical skin. Its zero-crossing reference surface still needs a matching positive child volume and conservatively verified child mass map. Simple nearest-donor embedding was ruled out (9,947 new points outside donor cells), and folded offset PLC trials remain rejected. No candidate native physical motion steps are claimed. Motion gates are predeclared in motion-acceptance.md.

Source base: 868486402da2ff0a208bf60c267df0928e2f91e0 plus the retained mass-ownership patch and this increment. The original dirty owner checkout and its index are untouched. Evidence distinguishes numerical consistency from physical motion qualification.
