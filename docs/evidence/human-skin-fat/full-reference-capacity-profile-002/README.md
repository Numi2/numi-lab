# Full-reference child mass capacity profile 002

The child mass mapping CLI now has an explicit opt-in full-reference capacity profile. The legacy profile remains the default.

| Profile | Nodes | Cells | Candidate edges | Augmentations |
| --- | ---: | ---: | ---: | ---: |
| Legacy default | 300,000 | 200,000 | 3,000,000 | 300,000 |
| Full reference, opt-in | 500,000 | 2,000,000 | 12,000,000 | 4,000,000 |

To opt in through CMake, configure with -DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=ON. That option applies the preprocessor definition NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1 to the child mass mapping CLI. Direct consumers of the capacity owner must use the same definition explicitly. Leaving the option off or the macro undefined selects the legacy profile.

The profile changes only bounded importer and sparse-transport workload capacities. It does not change source masses, material or donor/region barriers, density bounds, conservation tolerances, cell positivity, or the existing maximum transport distance of 0.019353 m. It does not expand native simulation/runtime limits.

CMake exposes four focused checks:
- numi-human-skin-fat-legacy-capacity-check
- numi-human-skin-fat-full-capacity-check
- numi-human-skin-fat-legacy-spatial-mass-transport-check
- numi-human-skin-fat-full-spatial-mass-transport-check

The staged regression compiled and ran the CLI, capacity-profile importer fixture, mass-mapping owner fixture, and conservative spatial transport fixture in both legacy and full profiles. Each completed with exit code 0. The fixtures check profile values, header caps, malformed and nonpositive cells, semantic labels, mass and source-donor closure, deterministic receipts, bounded distance, and candidate-edge/augmentation budget rejection. The exact receipt is checks.json.

The source-330 reader check loaded 28,632 parent tetrahedra and 13.939678963460464 kg from the pinned parent packet. It was read-only: child transfer was NOT_RUN. No full-volume transfer, TetGen run, native runtime, accepted motion, or physical admission is claimed.

The capacity profile is not evidence that a particular full-reference volume fits its limits.
