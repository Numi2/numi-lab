# Accepted rendered geometry audit 001

This is a post-run CPU audit of the existing MRVPack V2 exports. It did not run or advance simulation state.

Run root on the Mac mini: `/Users/n/numi-human-resting-evidence-20261005/cardiac-native-admission-002/`. The immutable run record is `invocation.json`; source HEAD was `a4334df01211f4c9c2991da98780ece0743ecf59`, the retained worktree diff SHA-256 was `0f74f989e261641ef23ce9fe48fc8f9282650ec813c3d015102152c1d1170aa9`, and the native executable SHA-256 was `9cbdfc443b659f73304206f704162c43d82b2bd4adb333c0a23e198729b8fbe0`. The device was Mac16,11 (arm64, macOS 26.6). The run accepted 3,000 steps at 2 ms with zero root assistance; `timing.txt` records 6.000 s simulated / 64.754 s launch wall (0.09266 RTF), while the native horizon timer reports 61.220 s / 0.09801 RTF. The exact launch argv, payload hashes, binary, libraries, metallibs, CSV traces, and continuous viewer recording remain beside the packs.

Four requested rendered state IDs were audited: 0 (0 s), 639 (1.2780000607 s), 2783 (5.5660002644 s), and 2999 (5.9980002849 s). For each pack and receipt, the whole-file SHA-256, header content hash, each section hash, and captured vertex-buffer hash agree. Each contains 2,000,912 finite world-space vertices, 9,591,711 indices, 861 primitives, and 861 instances. Indices and instance records are byte-identical across all four packs; primitive geometry/identity records are unchanged while bounds are recomputed from each captured geometry. The five lung lobes (IDs 305–309), diaphragm (311), ribs (44–47 and 56–59), heart/cavity identities (318–321), and organ identities are present at all four states.

| Presented step | Lung lobe mesh volume (ml) | Same-frame `lung_target_ml` (ml) | Mesh minus target (ml) | Minimum skin/bed gap (mm) | Vertices beyond 1 mm penetration |
|---:|---:|---:|---:|---:|---:|
| 0 | 3147.544400 | 2499.999944 | 647.544456 | -0.110552 | 0 |
| 639 | 3700.105027 | 3052.563407 | 647.541620 | -0.002474 | 0 |
| 2783 | 3457.757696 | 2810.216509 | 647.541187 | -0.003628 | 0 |
| 2999 | 3658.939194 | 3011.398483 | 647.540711 | -0.003710 | 0 |

The rendered lung envelope volume follows the same-frame gas-volume target with a stable 647.5407–647.5445 ml source-envelope offset (spread 0.00375 ml). That offset is expected from the registered source geometry: its receipt describes the lobe envelope as including tissue and gas space and explicitly says it is not FRC or gas volume. The full-skin minimum gap in the pack exactly equals the same-step `resting-surface-audit.csv` value.

Signed volumes computed from the exact rendered triangles also agree with the *same-frame* chamber target fields in `resting-surface-audit.csv` for all four mapped cavities: 318=RA, 319=RV, 320=LA, 321=LV. Across 16 comparisons, the maximum absolute relative error is 6.29e-7 (LV at step 2999, -7.9515e-5 ml). The compared fields are the same presented state, not a neighboring row from `resting-coupled.csv`.

An early exploratory check accidentally compared a rendered frame to the next accepted endpoint row in `resting-coupled.csv`; that comparison is invalid because the viewer shows the pre-dynamics state. It is superseded by the exact step/time-aligned surface-audit comparison above and is not used as evidence.

Cadence preflight was separately tested with a scratch-compiled native app object and the frozen build-001 libraries. Requesting unreachable presented step 1500 returned 1 in 0.717 s with `accepted geometry step is not presented by the native submission cadence: 1500`; the requested output directory was not created, and the frozen build-001 executable, libraries, metallibs, CMake cache, and compile database hashes were unchanged. This was a CPU-only validation, not a physical run.

This audit verifies exported geometry identity, finite positions, exact topology retention, full-skin contact gap telemetry, and rendered lung/chamber volume correspondence. It does not certify all organ interfaces, source self-intersections, cardiac wall correctness, clinical plausibility, or clinical validity; those remain separate geometry and physiology gates.
