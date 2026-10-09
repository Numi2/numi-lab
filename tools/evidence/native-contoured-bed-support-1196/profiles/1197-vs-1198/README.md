# Contoured-bed profile comparison (1197 vs 1198)

This is a read-only comparison of the retained, successful 20-second flat-bed reference run (1197 attempt 002) and contoured-bed smoke run (1198 attempt 002). Both used 10,000 accepted 2 ms steps and produced 1,250 native profile samples. The recorded launch arguments agree after normalizing output/movie paths and removing the single declared `--resting-bed-surface` argument. The runs share a base revision and loaded MetalRobo library/shader pins. They are not byte-identical builds: native executable hashes differ, and the source patch in 1198 includes the required orthonormal shading-tangent fix for the contoured mesh.

## Measured differences

- Native command mean: 130.872 ms to 177.342 ms per profiled eight-step segment (+46.470 ms, +35.5%). GPU time accounts for +46.459 ms of that increase; host-copy time rises by 0.0075 ms. Across 1,250 segments, summed profile GPU time is 162.895 s vs 220.969 s.
- Integrated body wall time: 185.684 s to 242.944 s for 20 simulated seconds (+57.260 s, +30.8%). Whole-run wall time is 192.673 s vs 250.408 s.
- Regular rendering is nearly unchanged: mean GPU render time 12.933 ms to 13.139 ms (+0.206 ms, +1.6%). Presentation mean is 54.477 ms to 58.273 ms (+3.797 ms, +7.0%).
- Full geometry export includes 4.25% more vertices and 4.87% more triangles in the contoured run. Mean export/audit wall time rises by 136.0 ms (+9.1%).
- Aggregate accepted-observer callback time rises 0.882 s over 1,250 callbacks (+6.2%). The support-impulse CSV phase contributes +0.083 s total, or about 0.067 ms per callback; presentation is the largest listed observer phase.

These logs time the whole native GPU command but do not separately time bed query/contact-plane update kernels. Therefore the GPU delta is associated with the contoured-bed mode and is not an isolated kernel-cost measurement. The two attempt profiles are not repeated controlled benchmarks, and different executable hashes limit causal attribution to the bed alone.

## Scope and limits

The contoured-bed smoke is a 20-second profile, not a long-horizon timing qualification. Its contact trace passed, but a later anatomy audit found 3,727 skin-target intersections across 26 surfaces and zero self-intersections. This profile comparison does not establish anatomy acceptance or authorize optimization. No GPU run, source edit, or optimization was performed for this comparison.

`profile-comparison.json` contains the per-run profiles, deltas, run metadata and input SHA-256 pins. `compare_profile.py` regenerates it from the retained logs and metadata. `checksums.json` records hashes for this bundle and confirms the pinned source inputs still match.
