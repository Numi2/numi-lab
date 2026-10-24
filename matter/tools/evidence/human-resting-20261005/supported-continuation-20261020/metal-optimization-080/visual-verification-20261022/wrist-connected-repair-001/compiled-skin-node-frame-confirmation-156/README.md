# Compiled skin/fat coordinate confirmation

The opt-in NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV path dumps the existing compiled initial Float32 nodes and masses once during source preparation. It does not update, deform, constrain, or add state to the simulation. Runtime physics and accepted-state behavior are unchanged.

Strict app compilation/linking passed. The complete-scene source preflight reached its intentional source-only stop in 8.2224265 seconds. The dump has all 7,233 nodes (2,411 each in outer skin, shared skin/fat interface and deep fat), global IDs 1666..8898, and total skin/fat mass 13.939678935082 kg. Detailed registered skin and 86 body poses are byte-identical to native source-preflight005.

All emitted nodes agree with the pinned D36 source-to-world rigid placement to less than 0.13 micrometres; see per-layer maxima and exact hashes in frame-validation.json. Subsequent exact collision checks should use these emitted coordinates because submicrometre rounding can change grazing witnesses.

This confirms coordinate ownership and preparation only. No physical step, skin clearance, full anatomy, sustained performance or physiological qualification is established by this run. The mixed-source reference anatomy provenance remains in the existing D36/source manifests; the dumped coordinates are inferred compiler geometry, not measurements of an individual.
