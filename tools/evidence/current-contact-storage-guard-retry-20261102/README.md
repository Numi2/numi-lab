# Repeated Human working-set admission

The optional Human contact working-set guard was checked only while constructing the Metal device buffers. A second encode attempt after rejected admission could bypass that check. The guard now runs before every encode, including retries. It neither prunes physical rows nor modifies forces, material equations or accepted-state tests.

The new regression was compiled and run against the unchanged preimage library first: it failed with `invalid working-set guard was bypassed on retry`. After the fix, native build, immutable shared-storage tests, grouping tests, and both physical surface-attachment fixtures passed on SSH Numi (Apple M6). Those fixtures retain the force, accepted deformation, bitwise rejection rollback and replay checks. The publication checkout also passed its own Objective-C++ syntax check.

The complete Human motion remains unqualified. The tests are small native fixtures, not a whole-body motion result. The 760-file execution-source manifest pins the dirty execution checkout separately from this narrow publication patch. Full-body trial 096 is a separate, unqualified capacity diagnostic.
