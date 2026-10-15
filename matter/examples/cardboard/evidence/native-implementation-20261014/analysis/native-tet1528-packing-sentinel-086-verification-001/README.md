# Packing verification supplement for overlay 086

This supplement verifies the selected trace's state/trace output layout without modifying the sealed overlay. Run:

```sh
python3 packing_sentinel_test.py --output verification.json
```

The check is source-bound to the sealed shader, checker, modified common material source, and snapshot 086 shared header. It verifies 32-state capacity versus the 15-state material writer, exact high/low/sentinel/trace offsets, disjoint destinations, host buffer size, and sentinel bit patterns.

The result is a static source and CPU packing-emulation pass. It is not a native Metal run. See [ERRATUM.md](ERRATUM.md) for the superseded defect pattern and its limitations. Overlay 001 remains unchanged. Its CMake patch has a separate integration issue: inserted CMake variables contain escaped dollar signs; root is correcting that in a new source overlay.
