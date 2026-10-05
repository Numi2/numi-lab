# Existing Metal equality solver optimization

All builds and execution ran on the SSH Mac mini, Apple M4 Pro. The complete
integrated scene with explicit rigid digits has 91 equality rows. Profiling
identified the scalar fallback above the 64-row threadgroup cache as a major
bottleneck: approximately 19.07 ms for factorization and 24.52 ms for finish
per sampled physical step.

The existing factorization now parallelizes independent rows in the same
device allocation for larger blocks. Its pivot search, row arithmetic and
FMA order are unchanged. The existing cooperative triangular solve also reads
larger factors from device memory. No new matrix allocation, CPU physics,
physical approximation, solver tolerance or additional readback was introduced.
Smaller blocks retain their threadgroup factor cache.

Factorization alone fell to about 1.14 ms and reduced total GPU time 29.76%.
The subsequent triangular-solve change lowered sampled finish time to about
18.57 ms. Together the changes reduced measured GPU time 39.53% over the
six-second integrated comparison. Final wall time was 119.966 s for 6 s
simulated (0.0500143 times real time). CPU asset preparation ran concurrently;
this is a local engineering comparison, not a general hardware benchmark.

The control, first candidate and final candidate are bit-for-bit identical
for respiratory/circulatory and surface CSVs, the initial pack, all three
accepted geometry packs, and terminal q/v state. Rejection and retry checks
passed with unchanged controller/physiology history. A separate 32-step
comparison preserved the ordinary 51-row source configuration exactly.
The existing skin audit results transfer only to those identical captured
geometry bytes. Known lung/rib and cardiac defects remain; these runs do not
qualify the final anatomy or continuous full-duration scene.

`manifest.json` binds compact evidence to original Mini paths and hashes;
gzip files are lossless. The baseline lives in adjacent `hand-native-032`.
Build invocation records explicitly identify the source018 CPU bundle clone,
changed Metal compile/link commands and executable rpath relocation. Source
snapshots 019/020 and all unretimed recordings remain on the Mini. The shader
changes are in the existing NumiHumanStand solver, not a separate runtime.
