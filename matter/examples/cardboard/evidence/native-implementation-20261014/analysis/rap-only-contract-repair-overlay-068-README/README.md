# Observer validation overlay 068

Base: frozen snapshot
`/Users/home/cardboard-implementation-20261014/snapshots/source-consistent-armijo-alpha-and-topology-capture-067`.

This isolated overlay changes only runtime diagnostic-environment validation,
the source-bound runner receipt path, and the Python diagnostic-history
comparator/tests/docs. It does not alter solver policy or physics.

The runtime now reports an invalid `NM_FEM_REGIONAL_RAP_CAPTURE_ONLY` value,
or a mode-1 request missing its selector/path or regional preconditioner,
before checking whether the selected control root is current. These messages
are observational and do not change solver state. The comparator verifies the
regional manifest and binary sidecars, selector binding, array packing and
hashes, and per-environment candidate identity digest. The source-bound runner
retains `timed_out=true` if process-group termination fails, records failed
signals, and preserves the original OS exception in the receipt. An interrupted
input scan is represented as unknown.

Validation from `work/matter/tools`:

```sh
python3 -m unittest test_cardboard_diagnostic_history_compare
python3 -m py_compile cardboard_diagnostic_history_compare.py cardboard_source_bound_run.py cardboard_source_bound_run_support.py
```

The runtime translation unit also passed an Objective-C++ `-fsyntax-only`
compile with the frozen 067 headers. No GPU or native physics run was used.

`changes.patch` is the portable unified diff. `source-overlay.json` binds the
changed files to their exact base and overlay SHA-256 values.
