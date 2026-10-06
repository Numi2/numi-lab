# Paper-law local residual scale diagnostic

This fixture tests whether FP32 generic material projection is accepting an implicit-state solution too loosely for the whole-board force balance. It is diagnostic-only and does not change the canonical materials.

Each copied `.nmatter` adds fixed `residual_scale = 10` and multiplies all 13 implicit residual equations by that factor. The physical roots, yield surface, elastic/plastic parameters, and global convergence gates are unchanged. Since the local solver still uses the same absolute residual test, the effective acceptance threshold in the original unscaled equations is approximately ten times tighter. This is an authored numerical-conditioning experiment, not a fitted material parameter.

`paper_fp32_residual_scale_trace.cpp` is a scratch copy of the paper checker’s FP32 bytecode trace driver. It compiles both canonical and scaled source through the same Matter compiler, reproduces the local Newton solve and 16-trial Armijo line search, and evaluates the canonical unscaled residuals at each scaled solution. It performs no GPU work.

The trace uses the established mixed deformation perturbations around `F11 = 0.994` for the liner and `F11 = 0.9975` for the medium. Results are in `fp32-local-trace.txt`: all four canonical and scaled local projections converge. Scaling takes one additional local Newton iteration in each case. Canonical residual maxima are 2.74e-5–3.60e-5; the canonical residual evaluated at scaled states is 8.35e-7–1.07e-6. The native assembled-board run is still required to test whether this local-accuracy change resolves the global force-balance stall.

## Reproduction

From `/Users/home/numi-cardboard-20261006`:

```sh
clang++ -std=c++23 -Wall -Wextra -Wpedantic -Wno-unused-function \
  -I matter/include -I matter/tools \
  /Users/home/cardboard-evidence-20261006/paper-residual-scale-10/paper_fp32_residual_scale_trace.cpp \
  build-cardboard/libnumi_matter_compiler.a \
  -o /Users/home/cardboard-evidence-20261006/paper-residual-scale-10/paper_fp32_residual_scale_trace

/Users/home/cardboard-evidence-20261006/paper-residual-scale-10/paper_fp32_residual_scale_trace \
  matter/materials/hajali2009_liner_hill_ideal.nmatter \
  matter/materials/hajali2009_medium_hill_ideal.nmatter \
  /Users/home/cardboard-evidence-20261006/paper-residual-scale-10/hajali2009_liner_hill_ideal_residual-scale-10.nmatter \
  /Users/home/cardboard-evidence-20261006/paper-residual-scale-10/hajali2009_medium_hill_ideal_residual-scale-10.nmatter
```

## SHA-256

- Canonical liner: `62f056400c7c0c42f0d831ffa8be9738ee666bd05b6d4513d7cfa82a5a32eb39`
- Canonical medium: `01507b68491e4d34255d65ed9ff14c17a08540c782fce1e92ea16d620a2007c0`
- Scaled liner: `c6cd258db8516a7ef3a3a8c1f2482c1a32e1e8b972dd07dbf8b4fb6a9e9d758c`
- Scaled medium: `d1665e8846f48bc366f0f3d50bb8eafb8c6d382dd0acf9d9c4f8f8848a3cfce5`
- Scratch driver: `d0ca522a9c3370d09ac94d93fd53ba8b89f263dc17ffd12fcbb9bcb464a0dda6`
- Compiler library: `d1d984663c87e7f6e23e327dfbe180ab5e3b9d99b558c384d52fa50d156a978a`
- FP32 trace: `44809e8fed1be46de669ab1acccd74b8421738f0bfeaf99edf7155f1886443ae`
