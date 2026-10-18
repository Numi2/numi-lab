# Deferred-audit microtick regression 227

Native Apple M6 run passed in 3.063 s. The unchanged 225 suite ran first; the 227 extension accepted the configured two-microtick step, rejected an expected-count mismatch with bitwise rollback, and rejected invalid configured counts 0, 3, and 512. Exact source, build, invocation, binary, runtime-library, shader, and result pins are retained here and in `../tests/`.

This is a focused adapter/runtime regression. It does not qualify the 0.5 ms shared-scene run beyond the measurements in `native-009/`.
