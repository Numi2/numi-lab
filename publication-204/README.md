# Numi Human hand: measured reference-pose trials

This archive is an **offline reference-pose construction checkpoint**. None of the candidates was admitted to the native physical scene and no physical step was run. It extends the baseline and provenance in [P202](https://github.com/Numi2/numi-lab/tree/a931e064681f211973b32047b52a0372918562f9/publication-202) and the remesh and layer-transfer checks in [P203](https://github.com/Numi2/numi-lab/tree/d02d6c5a4d1b2a48576508bcded12bc048ff8236/publication-203). It is not a complete anatomy result, a simulation qualification, or clinical validation.

## Pinned checkpoint

Candidate SHA-256 806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756 contains 21,212 nodes and 42,379 faces. Its complete rest-pose audit of 185 typed bone banks found 977 raw skin/bone triangle pairs, 651 immediate-parent/bone-triangle identities, zero exact nonadjacent skin self-pairs, and 125 sampled skin-node/bone inside identities. Those are different measures, not interchangeable counts. This surface checkpoint is not the currently admitted native volume.

The three largest target banks contain the following inside samples, measured by nearest fixed-bone surface distance:

| body/bone | inside samples | median depth | maximum depth |
| --- | ---: | ---: | ---: |
| 43/16 | 43 | 0.59293 mm | 1.77822 mm |
| 109/114 | 24 | 0.53683 mm | 1.86801 mm |
| 65/99 | 29 | 1.65300 mm | 2.07527 mm |

The archive contains the pinned source query CSV and the script generating these statistics. An inside distance is a displacement lower bound against a fixed bone; it does not prove a feasible repair.

## Measured trials

- **43/16:** a connected witness solve stopped at its iteration cap. A saved direction rebased onto 806 and a targeted layer correction improved one already-negative fat-cell volume but left it negative. The trial increased exact target crossings from 247 to 249 raw and 149 to 151 mapped. It remains rejected.
- **65/99:** a connected witness fit passed its shape check at 16 of 17 backtracks but failed the local layer check at every one. The alpha-1 diagnostic increased target counts from 148 to 152 raw and 84 to 87 mapped. A second tiny fit passed local layer nonregression but left all target identities and all 29 inside samples unchanged. Both remain rejected.
- **109/114:** five initial position trials kept exactly the same 184 raw target pairs and 119 immediate-parent identities. The largest created four exact skin self-intersections. The quarter-step candidate e09d08936361dbfab4d6f92ecb5fcde3bfbeaeb1ac11a26f2e01fa3eeee18771 was checked against all 185 banks: the sets of raw and immediate-parent pairs are unchanged, self-pairs remain zero, and all 125 inside identities remain. Among the 24 target inside points, the largest depth improvement was 0.005308 mm and the largest worsening was 0.00001393 mm. It is a small private intermediate, not a cleared intersection or admitted repair.

The quarter-step's fixed-offset layer check raised strict changed-face failures on previously bad faces. A separate comparison found zero added or removed Boolean authored-orientation, conditioning and nonpositive-volume flags globally; some previously poorly conditioned positive fat cells became slightly less well conditioned. This does **not** make the layer valid: the baseline diagnostic includes two nonpositive Float32 tetrahedra per layer and many authored orientation failures. No conservative 806-to-native volume and per-cell mass ledger has been established.

The raw-identity comparison helper is tested with equal-sized sets containing a replacement pair. It reports added/removed identities when face identities are unchanged and explicitly reports them unknown for changed face domains; changed topology still needs independent ancestry validation. The source and test are included.

## Execution and provenance

All work represented here ran on SSH Numi: Apple M6, 12-core GPU, 16 GiB unified memory, macOS 27.0 build 26A425, arm64. The Metal compiler path was /Users/numi/Library/Developer/Toolchains/Metal.xctoolchain/usr/metal/32023/bin/metal; these trials were geometry preparation and did not run a Metal physical simulation. The scripts pin source hashes and use the retained geometry slot. The original absolute paths must be restored or remapped to reproduce them.

The registered reference assembly includes BodyParts3D FJ2810, © The Database Center for Life Science, CC BY 4.0 International: [license](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html); MyoSim is Apache-2.0. Mixed sources are not measurements of one person. The evidence branch does not modify product main.
