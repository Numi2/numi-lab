# Asymmetric transfer CPU calibration receipt

This directory preserves the completed CPU-only C1 source-curve calibration and the reported exact-ReLU corner failure. The C1 run exited 0 with 2,350,688,803 checks and the full 278 selected marker centers. It is a same-publication calibration, not held-out validation, native Metal execution, or physical board qualification.

- `c1-width3-cpu-calibration.stdout.txt` is the complete checker output.
- `c1-width3-cpu-calibration.stderr.txt` is the captured stderr (empty on the successful run).
- `legacy-exact-relu-corner-failure.json` preserves the partial reported failure values and marks the missing original raw receipt/full state explicitly.
- `run-manifest.json` binds the executable, checker source at seal, catalog, all four model assets, source CSVs, and receipt hashes.

The retained marker residuals and three-width sensitivity values are summarized in the manifest and detailed line-by-line in stdout. Compression and multiaxial transfer remain assumptions; no physical damage/folding qualification follows from this CPU check.

The 23-test source-data package suite is captured in `source-data-tests.stdout.txt` and `source-data-tests.stderr.txt`; it passed.
