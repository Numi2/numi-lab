# Persistent anatomical wall contact and shared FEM/MPM map

Base: b51a202241eb54e6393870dbefbc06a6c377ebd2. Executed only on SSH Numi. No visual verification.

The previous wall selector changed only a cooked world. The complete Human RestingRun copies the authored source and recompiles it, which lost that selection. Its earlier direct Metal predicate test remains valid for eligibility, but its body launch did not establish the final live mask.

ObjectSource now carries sorted exterior face exclusions using existing local tet/opposite-corner IDs. The compiler validates and emits the existing contact flag, including allocation-independent physical fingerprints. The anatomical wall authors its exclusions before cooking; read-only audits verify both preparation and the final live world. No wall geometry, reference operators, mass, cavity topology, extra blood state, or per-step CPU physics is added. Explicit nonadaptive MPM can coexist with the complete compact immutable FEM map. Adaptation, mutation and spare FEM topology remain rejected.

Checks:
- Existing static-map suite: 44 CPU checks, including selector validation, package roundtrip, different selectors' physical fingerprints, default allocation independence and mixed FEM/MPM layout.
- Actual registered stable-ID-9 wall: independent copy/recook preserves 384 outer contact faces, 672 exclusions, all 1,060 canonical rows (including 4 nonwall rows), 532 nodes, 1,584 tets and 0.00705926 kg.
- All 21 library translation units depending on matter.hpp rebuilt to avoid stale ObjectSource ABI; native application rebuilt and linked.
- Native mixed FEM/MPM Runtime.initialize and snapshot: PASS. Five FEM nodes/two tets/six faces and one MPM particle retained. No physical steps; this is initialization qualification only.
- Complete native scene: one 0.25 ms accepted step, exit 0, 54.007 s total launch wall time. Final runtime confirms 384/672 wall eligibility; coupled CSV equals prior baseline byte-for-byte. This short run establishes integration, not sustained performance or physiology qualification.

Limits:
Broader force qualification remains open. Separate registered organ and pressure-loaded wall approach fixtures reject during vertex-triangle CCD certification. Their separated controls and injected-rejection rollback pass. An eligibility test, a zero-contact control, and this one-step body regression must not be called accepted anatomical contact-force transfer or complete tissue-mechanics qualification. No clinical or material-accuracy claim is made. The passive effective-material assumptions and previous evidence remain in force.
