# Invocation amendment

The first chunk-8 invocation failed before simulation because state traces require chunk 1. Preserve its stderr and empty output. Retry the matrix without a state-trace option for all cases, using native summary counts. This changes the observation method, not the acceptance criteria or fixture.
