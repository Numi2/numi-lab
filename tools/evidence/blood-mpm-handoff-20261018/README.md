# Blood MPM handoff

The three tested source files are published at their canonical paths: matter/include/numi/matter/human_respiration.h, matter/src/human_respiration.metal, and matter/tools/human_resting_runtime.hpp. The package manifest retains their exact hashes under patchroot/; the duplicated copies are omitted here.

The test proves an MPM receiver transaction using a source-seeded external inventory: activation, mass/species representation, injected rejection rollback, same-context retry, accepted-state equality against uninterrupted control, and snapshot restore. It does not prove live CVSim outlet wiring, whole-body circulation-to-MPM integration, or a resolved free-surface liquid model. The external inventory remains the single owner; MPM represented mass is a subset.
