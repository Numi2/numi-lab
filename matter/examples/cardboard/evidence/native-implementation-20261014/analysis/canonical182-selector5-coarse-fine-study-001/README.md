# Source-182 RAP plus selector-5 graph correction: CPU emulation

This folder records a bounded CPU study using the selector-5 production graph layout and gather policy with source-182's selected-generation operator and selector-3 RAP capture. It is algorithmic CPU emulation only. Source 182 does not contain a selector-5 FEM patch-assembly or graph-factor capture.

The graph map uses source-065 `buildRegionalGraphSchwarzLayout` defaults (256 nodes per patch, anchor stride 128), rebuilt from the retained source-182 tetrahedron global IDs and candidate fixed mask. The Python reconstruction matches the retained source-065 exporter for every layout array, global ID, and FNV-1a checksum: 882 nodes, 1,920 tets, 846 free nodes, 36 fixed nodes, 13 patches, 2,053 patch-node entries, 6,159 DOFs, and 4,718,637 local matrix floats. The canonical overlay byte encoding independently recomputes topology SHA-256 `52d495969f0220838a87b2dd7984961265fc0e6b1891f2d3e157b2ef17ae7511` and free-mask SHA-256 `dae5a5047292398f5a4ebfae0f961daa5a76ec53af59bab106f326de08aeefd8`. Source 182's own RAP manifest predates those topology fields and has no selector field; selector-3 provenance is established by its retained source/layout lineage, not inferred from a missing manifest field.

The emulated action is `c = RAP^-1(P^T r)`, then local right-hand sides `r - AP c`, FP32 principal-block solves, nodewise sum of patch corrections divided once by exact overlap coverage, and `P c + fine`. The layout and gather ordering match the production helper/kernel. Coarse arithmetic uses a deterministic FP32 approximation to the Metal `simd_sum` tree; it is not hardware-bitwise. Local patch matrices are extracted from the independently audited CSR A and factored in SciPy, rather than assembled by the native selector-5 FEM path. Outer restarted GMRES and true residuals use the retained CPU utility and audited A.

## Final run

The final `run-006` used Python 3.10.12 with BLAS thread limits set to one:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 study.py
```

It used restart 16, a 256-iteration cap, and a true relative tolerance of `1e-8`. The run reached `8.572800977143566e-09` after 255 preconditioner applications and 526 A matvecs in 26.65 seconds. The local LU probe's maximum relative residual was `8.04e-7`; the minimum relative U-pivot was `2.49e-4`. The modeled numeric-array + factor-workspace + GMRES total was 60,358,208 bytes (57.56 MiB), below the 1 GiB numeric study cap. This is not measured RSS and excludes Python object/interpreter and SciPy/BLAS internal workspace; the exact component model is in `run-006/resource-estimate.json`.

The ordered attempt history is retained under `run-001` through `run-006`. Runs 001–003 are setup failures. Run 004 completed but did not converge (relative residual `0.9935469408593788`) and its probe used NumPy matmul that emitted divide-by-zero/overflow/invalid warnings; those warnings remain unresolved and are not interpreted as benign. Run 005 corrects a real axis-indexing error found in run 004, replaces diagnostic probe matmuls with scalar `math.fsum`, and converges with a deterministic flattened FP32 coarse reduction. A final fidelity check found that production computes one xyz dot per node before its 32-lane reduction, so run 006 applies that grouping, explicitly rejects nonfinite probe values, and converges without warnings. Run 004 remains a superseded failure; runs 005 and 006 are both preserved as CPU algorithmic emulations.

`study.py` refuses to overwrite its assigned output directory. For a new attempt, use a fresh copy and change only its `OUT` run-directory constant; preserve each attempt under its own `run-NNN` directory.

## Reader for the later topology-bound capture format

The separate v2 capture reader consumes the source digest fields introduced by `rap-only-topology-bound-capture-overlay-001`. Its CPU tests are:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/selector5-regional-capture-reader-v2-001/test_reader.py
```

Eight tests pass. The reader requires exact binary candidate fixed tags and scalar-integer NPZ node counts, recomputes the producer's canonical digest bytes, and rejects same-count alternative topology. No native selector-5 capture was available, so the reader has not been exercised against a real selector-5 capture in this work.

## Limits

This does not demonstrate a native selector-5 preconditioner run, native selector-5 local matrix/factor parity, Metal arithmetic or performance, successful physical stepping, material calibration, or physical cardboard behavior. Source-182's selected Newton 23 is a linearization diagnostic from a native run that later failed at Newton 63.
