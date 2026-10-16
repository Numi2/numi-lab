# Native 464 captured-state CGLS preconditioner audit

This CPU-only experiment applies 64 steps of right-preconditioned CGLS to the
source-151 capture's environment-0 free mechanical system, using the 419 FP64
CSR matrix and the matching captured base RHS. It compares identity,
node-local 3×3 inverse diagonal blocks (`block3`), and the unaugmented base
selector-5 graph/RAP map `S` with its algebraic transpose `Sᵀ`. The output
reports true residuals recomputed from `b-Ax`, relative residuals, mapped
correction norms, and `bᵀAx` at iterations 8, 16, 32, and 64.

At 64 steps, FP64 relative true residuals were identity `0.2296676`, block3
`0.0884539`, and base graph `0.5790494`; CPU-FP32 arithmetic gave identity
`0.2356153`, block3 `0.0897914`, and base graph `0.9234037`. All six runs
reached the fixed 64-step cap with finite checked vectors and no tolerance
based early stopping. These are bounded algebraic outcomes for this captured
linear system, not native solver behavior, convergence qualification, or a
physical result. The block3 inverse uses the 3×3 diagonal blocks of the
captured 419 CSR, and its transpose uses each inverse block's transpose.

The graph `S/Sᵀ` implementation matches the source-151 CPU equations on two
fixed-seed directions with relative L2 differences below `1.1e-13`; the
independent transpose dot-product check has relative gap `1.37e-13`. The
captured FGMRES state has its completed-tail output latch set. The benchmark
clones the capture object with that output latch cleared only to evaluate the
underlying source-defined *base* equations. It does not establish which map
FGMRES consumed; the augmented consuming restart generation is absent and the
source-151 audit explicitly refuses that claim.

The matching state/RHS/CSR join is bound through audit 464. That regional audit
also retains an entrywise graph-matrix `1e-9` absolute gate failure: max error
`7.3391e-8`, even though Frobenius relative errors are below `8e-8`. The graph
CGLS results must therefore not be taken as a qualification of patch matrix
agreement. CPU-FP32 uses NumPy/SciPy arithmetic and captured float32 factors;
it is not Metal arithmetic emulation. The installed NumPy 2.2.5 emits three
RuntimeWarnings for an ordinary finite `2646×26` matmul control; the control
and all recorded CGLS result vectors are finite. The six-run warning counts
are recorded in `results.json` rather than silently treated as nonfinite data.

Reproduce on one CPU BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/home/.pyenv/versions/3.10.12/bin/python3 run_cgls_preconditioners.py \
  --output results-reproduced.json
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_results.py
```

The full input/source hash bindings are in `manifest.json`. This audit is not a
native CGLS or selector-5 action replay, does not prove augmented-map
consumption, and makes no solver, physical, performance, or shipping claim.
