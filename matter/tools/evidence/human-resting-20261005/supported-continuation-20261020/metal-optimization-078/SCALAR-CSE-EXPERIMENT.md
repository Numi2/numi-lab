# Existing scalar-CSE activation

Baseline 200c7217. Current Material.cacheExpressions defaults false. First executable change enables the already implemented bounded exact scalar-expression reuse at Human source compilation, opt-in NUMI_HUMAN_MATERIAL_EXPRESSION_CACHE=1; constitutive equations/precision/acceptance unchanged.

Prediction before paired runs: at least 2x median four-root body throughput with existing 1% endpoint gate, assuming repeated expression work dominates. This prediction may fail; do not infer a gain from source counts. Stage timing remains a separate instrumented check.

First pair uses same native binary/shader, fixed existing 073 input/assets/config, four 0.125ms roots, flags differ only cache compilation. If parity passes, run two additional matched pairs. Keep uninstrumented control; do not count startup as body throughput. This is a release-transient numerical/performance test, not supported-rest or integrated 0.1x qualification. Preserve failures. Default remains off until measured and qualified.
