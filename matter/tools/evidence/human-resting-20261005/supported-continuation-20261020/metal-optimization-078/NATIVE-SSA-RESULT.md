Native exact-SSA material specialization: three matched four-root release-transient mechanics comparisons passed the retained 1% endpoint and FEM/fascia audit gates. Median speedup versus scalar expression caching is 1.555921361x. Native body times: 19.171084166, 18.837416792, 18.852991750 seconds for 0.0005 simulated seconds. Best measured body throughput: 0.00002654x. This remains far below 0.1x and is not supported-rest/end-to-end qualification.

The misleading directory name global-icb-native-base-1 is a NATIVE CANDIDATE (its invocation environment has NM_FEM_NATIVE_TANGENT_BUNDLE=1). Actual controls are global-icb-expression-base-1/2/3. Both bytecode bundle implementations were slower than the scalar cache control and remain disabled.

Cold process startup was 365.06 seconds for the first native candidate and 216.21 seconds for the first same-build scalar control; the next native process took 30.26 seconds. Initialization is reported separately, not hidden in a steady-state claim.

GPU stage-only observation of one root: assembly 0.6311s, linear 1.6512s, globalization 0.6571s. Work telemetry: 30 active Newton begins of 96 encoded; 1693 active Krylov columns over 108 working cycles of 384 encoded. These measured domains motivate complete-Newton-tail suppression and preconditioning work. Timing instrumentation changes encoding and is not multiplied into a predicted speedup.
