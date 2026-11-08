# Installed catalog input binding: non-G1 workflow

At exact unified source `d9db7f9f30f75c9d54432fee2ad89d0e8f3ef26d`, an
incremental configured build recompiled the modified Swift consumers and
installed a separate package. Installed and source selector hashes matched.
`plan.json` pins the installed executables, shader, selector, inputs, fixture,
and Python executable. The relocated WorldPack CLI help ran successfully.

The synthetic authored Franka WorldPack ran one training update and matched
selection on M6 at 2026-11-08 00:01:56.341–00:02:01.319 UTC, exit zero.
Inputs were unchanged during the run; the incumbent remained selected and no
lock or matching worker remained. Actual metadata included the correct world
hash. The non-humanoid screen was unknown and refused auto-load before,
during and after a copied same-path WorldPack mutation and a missing-file
check. Since this screen is unsupported for that world, those four identical
outcomes alone do not prove changed-input invalidation. The separately retained
CPU Python-producer/Swift-consumer check shows a synthetic supported-screen
status changing from passed to unknown for same-path replacement, removal,
and unresolved dependency closure.

This was not a clean-checkout or full clean-build test: the source was a
`blob:none` sparse partial checkout with 117,183 skipped tracked paths at
inspection, and build artifacts were incrementally reused. The 119,164
tracked paths include substantial historical assets; availability and build
cost from a new checkout were not tested. Environment evidence is recorded
in `build-environment.txt`: the active developer directory was Command Line
Tools, with Metal available via a separate installed toolchain. Training used
an existing Python 3.13 MLX/NumPy environment; the CMake cache's dedicated
MLX test interpreter was unset. Default Window visual assets remain missing.

Recorded file hashes do not cover all shared libraries, Python packages, OS,
environment variables, or transitive inputs, and do not make simultaneous
mutation atomic. Moving or removing evaluated input files makes status unknown.
The smoke is not a claim of physical accuracy, learned control, visual UI,
replay, clean-checkout reproducibility, or performance.
