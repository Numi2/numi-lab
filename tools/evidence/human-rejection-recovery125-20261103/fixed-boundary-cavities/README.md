# Fixed-boundary skin/fat cavity conditioning
Initialization component and **failed integrated trials**, not native-motion qualification.

The authenticated NPZ uses the existing proposal contract. The compiler composes disjoint
complete 3-to-2 edge stars, preserves Float32 vertex coordinates, verifies exact oriented
boundary/positive dyadic volume and existing native quality, and rebalances source mass
flows inside the frozen 0.187444 m transport radius. Source donor mass and source first
moments remain conserved; individual nodal lumped masses and their first moment can change.
This changes the FEM approximation and requires subsequent force/strain/motion qualification.

The full initial 42,727-cavity batch failed the bounded exact-arithmetic range check in one
cavity (015). The 42,726 geometry-screened batch identified 1,072 transport-radius failures
atomically (016). Filtering those witnesses produced 41,654 cavities. Reproduction selected
identical NPZ arrays and identical SHA-256:
a8e42f2da474a7297dea43c06414e8345f93f3a2ce63b72ccbf62bd76b61e22c.

Native import (017) passed geometry/mass checks: 1,495,294 -> 1,453,640 cells,
125,416 affected nodes, maximum donor mass error 4.3520742565306136e-14 kg.
Maximum nodal mass change 0.0028186855141347431 kg; nodal mass L1 change 0.1883242202694653 kg.
Source donor moment error <=2.3314683517128287e-15 kg m. Lumped nodal moment change:
[9.5261802914887828e-7, 3.2970332703539972e-7, 3.1060013774578181e-8] kg m.

An independent whole-topology audit found exactly the same 360,422 exterior triangles,
zero extra/missing triangles and zero nonmanifold faces. All coordinates are fixed.

**Motion failed:** 017 accepted only 3 x 15.625 microseconds, then rejected root 3,
Newton 1, contact candidate 3349706, primitive pair [299158,299707], VT direction 3/vertex 0.
The retained earlier topology reached 8 accepted steps in matching configuration. This
batch is not an accepted replacement for the scene. Do not call import success sustained
motion, supported rest or physiological qualification.

The standalone C++ component check passes heterogeneous mass, shared-boundary cavities,
exact geometry, frozen radius, ancestry and 14 atomic-negative cases. Its initializer is
opt-in through NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ and matching SHA-256. No physics,
force, determinant tolerance, clearance, residual gate, or rollback rule is disabled.

All execution was on SSH Numi, Apple M6 / 16 GiB. Per-trial plans identify source assets,
commands, configuration, source revisions/hashes and binary dependencies. The 017 source
archive plus dirty patch retains the tested implementation; other trial manifests keep
their exact pins. Large imported source anatomy is referenced by its immutable existing
receipt, not copied into this evidence directory.

Replay construction:
python replay_cavity_filter.py --proposal conditioning-shared-cavities-001/fixed-boundary-volume-proposal.npz --geometry-screen cavity-native-geometry-screen-001.json --native-transport-log EXTRACTED_NATIVE_016_STDERR --source-cell-count 1495294 --output FRESH_DIRECTORY

The C++ screen is a reproduction tool for trusted prepared inputs, not a public importer.
Native composition remains authoritative. Failed and excluded cavities are retained.

The large construction report is losslessly gzip-compressed; decompression was verified byte for byte. No source or runtime state was removed.
