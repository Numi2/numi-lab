# Shared refined-layer labels and connected skin transfer

The refined paired skin/fat mesh no longer has six tetrahedra for every current surface face. The optional explicit-topology path validates parent/child connectivity and material/source ancestry, then prolongs the independently computed original donor and regional basis labels. The legacy identity path is unchanged. This permits local mechanical refinement without losing physical source labels.

The optional connected-face search expands the existing transfer seed by up to three edge-connected rings, preserves every legacy seed, restricts only newly added faces by donor identity, and uses stable source-face identities for deterministic ties. Default ring one retains the old rows byte for byte. GPU transfer row layout and physical forces/state ownership are unchanged.

Strict compile/run tests passed on SSH Numi against main 62afeeb2. The parent-label test compares all 28,632 synthetic child labels with the independent legacy builder on 27,714 parents and checks the exact frozen candidate's dense execution-face IDs. It is a label contract test, not a geometry or physics qualification. The ring test checks default parity, seed preservation, connected support, stable ties, partition unity and invalid bounds.

The D19 complete-scene mechanics diagnostic accepted eight 125-microsecond steps (1 ms total), minimum tissue Jacobian 0.96519. It begins from an unbalanced release and does not qualify supported rest, range of motion, physiology, or the requested long runs. Its unrequested replay was incorrectly labeled failed; the reporting-only correction is separately published in 329b01b8.

D19's final manifest was rewritten after launch and the original launch-manifest bytes were not retained. Source, binary, asset and configuration hashes are individually retained; this is not an exact launch-manifest claim. D20 uses exclusive immutable launch snapshots and tests the combined ring-two transfer. Neither unadmitted candidate is promoted as the default anatomy by this change.

All implementation/builds and native diagnostics ran on SSH Numi. No images were opened or used for iterative tuning.
