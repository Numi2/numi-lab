# Contact-iteration diagnostics on the Mac mini

These are 24-second integrated native runs on the Apple M4 Pro, using the same
source023 binary and unchanged full-chain003 anatomy and physiological inputs.
They are engineering evidence, not final whole-human acceptance. All computation,
audits and recording ran on the SSH Mac mini; other CPU anatomy work ran concurrently.

| Existing contact sweeps | Run | Wall seconds | Real-time factor | Late COM slope x/y/z (mm/s) |
| --- | --- | ---: | ---: | --- |
| 16 | native049 | 272.370 | 0.08812 | −0.07793 / −0.02511 / −0.00153 |
| 32 | native050 | 298.514 | 0.08040 | −0.04459 / −0.02370 / +0.00102 |
| 64 | native052 | 349.050 | 0.06876 | −0.00728 / +0.00845 / +0.00735 |

The slopes fit retained accepted samples from 18 to 24 seconds. Increasing sweeps
reduces the displacement rate but does not establish static equilibrium or a
five-minute drift bound. The 44 physiological CSV columns are bit-for-bit equal
across these runs; only the three contact columns change. Every run completed
12,000 steps with unchanged inputs, four completed breaths, 28 complete filling /
ejection cycles, and no root assistance. Peak blood-volume accounting error was
0.007917 mL. These numerical properties do not imply physiological or anatomical
qualification.

Exact whole-skin self-intersection checks found zero pairs in native049 at steps
4991, 9311 and 11999, and at step11999 in native050 and native052. Native049 passed
the former 18.622-second mesh failure point, but its weak Float32 triangle margins,
stale derived pleura, lung/rib intersections and ventricular-wall self-intersections
remain blockers. Source and native passive-organ neighbor censuses are retained;
their geometric contacts require component-specific anatomical interpretation.
The two zero-area taenia mesocolica triangles were not omitted from the audit.

Failed native048 is retained: its stale nested cardiac output hash was rejected
before any physical step. Full-chain003 refreshed that hash only after checking
the relevant cardiac arrays were unchanged. The preliminary passive audit's
incorrect receipt filename failure and corrected driver are also retained.

The native049 recording contains 376 strictly ordered image frames over 274.21
wall-clock seconds and was not retimed. The Mac mini had no graphical desktop;
the native Metal viewer rendered and recorded offscreen, so interactive window
operation remains unverified. Exact movie, pack and image hashes and retained
paths are in `manifest.json`. The source023 build provenance is in the neighboring
`limit-lanes-047` evidence; its relocatable build must not be identified by its
directory's git HEAD alone.

Human revision `9f9a740` exposes the existing solver setting as
`numi human resting-run --contact-iterations 64`; it leaves the default unchanged.
All 15 launcher tests passed on the Mini. Large JSON evidence is losslessly
gzip-compressed; the manifest preserves the original and retained hashes.
The manifest SHA-256 is
`6b61cb94ea13543f520f730a66a845a913ad41e14a83de635546c83f0c3f21da`.
