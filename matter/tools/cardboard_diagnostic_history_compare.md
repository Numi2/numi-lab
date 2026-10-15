# Cardboard diagnostic history parity

`cardboard_diagnostic_history_compare.py` compares a sealed diagnostic-enabled
native attempt with a diagnostic-disabled control. It admits a pair only when
both source-bound receipts executed, inputs remained unchanged, neither timed
out, and the archived source, executable, Metal libraries, build provenance,
physical arguments, and non-diagnostic environment match. The only permitted
environment differences are the explicit FEM Newton-trace selector, the
operator-FD selector, and its per-attempt capture path. The capture must remain
inside that attempt directory.

The comparator walks the complete `output/` trees recursively and records a
SHA-256 and byte length for each file. It also compares the actual bytes, so
nested conditioning snapshots, material history, meshes, OBJ files, CSVs,
manifests, and result/certificate output are included. Diagnostic capture files
outside `output/` are intentionally excluded. The JSON report is created
exclusively; an existing report is never overwritten.

Example:

```sh
python3 matter/tools/cardboard_diagnostic_history_compare.py \
  --diagnostic /path/to/diagnostic-attempt \
  --control /path/to/control-attempt \
  --output /path/to/new-report.json
```

An identical output tree establishes diagnostic parity only. Each attempt is
reported separately from the comparison. A failed or incomplete run remains a
failed or incomplete run, even when every output byte matches; parity never
promotes it to convergence, physical validation, or qualification.

## Retained comparison 057/058

The CPU comparison at
`/Users/home/cardboard-implementation-20261014/solver/diagnostic-history-compare-057-058-002.json`
matched the source snapshot, executed build, 693 archived input hashes, and
all physical argv entries except `--output`. The only environment differences
were the three allowlisted diagnostic selectors/capture path. Both receipts
executed, preserved their sealed inputs, and finished without timeout; their
timeout ceilings differed (1800 s and 600 s), but neither ceiling was reached.

All 20 production output files were byte-identical, including the rejected
conditioning arena. Both attempts exited 2 with status `failed` after one
accepted step of the 27-step declared protocol; both set
`physical_validation: false`. The result is therefore `not_qualified`. This
comparison records that diagnostic capture did not change this failed attempt's
production history; it does not show that the solver completed the cycle.

The unit tests inject changes to physical argv, input archives, unapproved
environment variables, nested binary outputs, timeout receipts, and capture
paths. They also verify that identical failed runs remain unqualified and that
report creation refuses to overwrite an existing file.
