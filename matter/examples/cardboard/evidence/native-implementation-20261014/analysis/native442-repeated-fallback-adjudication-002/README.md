# Native442 repeated-Cauchy audit v2

This CPU-only audit corrects the telemetry classification in adjudication-001. The earlier report counted every `base_cauchy` event as an executed fallback. This version uses source137 state gates and per-environment record comparisons; it preserves the original run, receipt, and report.

Reproduce with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/native442-repeated-fallback-adjudication-002/audit.py
```

The script parses the immutable native442/native435 logs and receipts, verifies input-file hashes and first-control outputs, checks the :all Cauchy trace against the primary pass, and emits strict JSON (`allow_nan=False`). It performs no native execution and launches no GPU work. See `report.md` for the concise verdict and `report.json` for per-environment classifications, source hashes, and retained files.
