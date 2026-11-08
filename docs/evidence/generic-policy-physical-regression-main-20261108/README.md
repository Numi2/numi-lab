# Imported-policy physical regression gate on main

A higher-reward synthetic imported-task candidate with one extra physical
contact termination in sixteen environments was selected while the same
decision recorded an increased termination rate as a regression. The retained
counterexample is before.json. The selector now requires the reported physical
failure rate not to increase before an imported-task candidate can advance;
a failed physics step remains a separate veto. The candidate artifact remains
available for research.

Native timeout reason 4 and food-consumed success reason 7 are not counted as
physical failures. This port also reads native reason counts when an explicit
timeout count is absent. These rules are limited by the evidence and reason
ABI; arbitrary custom-task termination semantics and missing or malformed data
are not qualified.

The unmodified main suite passed 27 tests. The first port run failed because
the command-level test used subprocess without importing it; that error is
retained. After adding the import, all 31 tests passed. Tests include
WorldPack and URDF failures, timeout and known-success cases, and the real
selector CLI using synthetic evaluator output and policy bytes. They do not
establish physical-policy quality, graphical UI, or platform qualification.
