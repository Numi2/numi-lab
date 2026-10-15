# Synthetic damage/cohesive transaction fixture overlay

This source-bound overlay adds a small native checker, an explicit synthetic parameter JSON file, and a Matter CMake registration patch against frozen source snapshot 073. It does not calibrate the corrugated-board recipe or qualify a physical box.

The mesh comprises three 10 mm triangular prisms, each 1 mm thick and split into three tetrahedra. The 18-node, 9-tetrahedron world has liner, finite elastic glue, and medium regions. The liner and medium retain their respective source Hill programs and are remapped to the retained-matrix 30-state damage overlays. The middle three tetrahedra use the exact `starch2007_finite_glue_elastic.nmatter` asset. Two duplicated coincident interface node triples form independent dormant mortar patches with stable IDs 701 and 907. Both interfaces use the synthetic mixed cohesive law in the parameter JSON; those law values are test-only.

The native CTest accepts the setup endpoint, activates each interface in its own accepted transaction, applies three small top-surface grip displacements, unloads, checks damage-history advancement, submits a deliberately singular endpoint and verifies physical-state rollback, then restores the accepted checkpoint in a fresh runtime and compares the next accepted continuation. Every native transaction logs status and certificate values. Before a native assertion fails, the checker emits a full exact readback snapshot in little-endian hex for FEM nodes/fields/topology, material state, schedulers, cohesive faces, contact histories, and rigid/other snapshot arenas.

The checker hashes the bytes of the three `.nmatter` assets and supplies those hashes to the damage builder. It requires the parameter JSON bytes to match the compiled synthetic parameter record and uses that file's actual SHA-256 for the damage set and each explicit synthetic-set binding. The parameter record names the fixture set/bindings and reports the synthetic evidence class. Source and parameter hashes are printed at invocation and are recorded in `source-overlay.json`.

## CPU verification

Strict Objective-C++ syntax, manual link against the snapshot-073 Matter archives, CPU admission, a CMake build harness, and its CTest passed. Receipts are under `receipts/`. The CTest harness imports the prebuilt source-073 archives and exercises the CPU-only invocation without starting native transactions:

```sh
cmake -S cpu-ctest -B /tmp/cardboard-damage-cohesive-transaction-074-cmake \
  -DNUMI_SOURCE_ROOT=/Users/home/cardboard-implementation-20261014/snapshots/source-reconciled-angular-fixture-and-probe-cli-073 \
  -DNUMI_BUILD_ROOT=/Users/home/numi-cardboard-20261006/build-cardboard-provenance-073 \
  -DBUILD_TESTING=ON
cmake --build /tmp/cardboard-damage-cohesive-transaction-074-cmake \
  --target numi-matter-cardboard-damage-cohesive-transaction-check -j2
ctest --test-dir /tmp/cardboard-damage-cohesive-transaction-074-cmake \
  --output-on-failure -R matter.compiler.cardboard_damage_cohesive_transaction
```

`overlay/matter/CMakeLists.patch` registers CPU and native CTests in Matter's CMake. The native test is pending a fresh root-owned snapshot and run. The scratch CMake/CTest harness is a link and CPU receipt, not a native execution.

A tampered parameter JSON was rejected before compilation/admission; its exact error is retained in `receipts/parameter-mismatch.stderr.log`.

One earlier invocation intended to run `--cpu-only` reached native execution because the pre-fix CLI branch checked the wrong argument count. It ran 10 native transactions and passed before the final provenance/logging changes. It is explicitly recorded as exploratory, mutable-source evidence in `source-overlay.json`; do not treat it as the frozen native qualification. The corrected four-argument CPU branch is the one exercised by the sealed CPU receipts.
