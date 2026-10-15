# Common-step FP64 diagnostic sweep

The two larger deformation-norm steps 1e-6 and 3e-7 were declared before execution and each uniformly replays all 1,728 rows from the same rejected source101 capture, with the same accepted histories, nominal directions, timestep and v3 projection gates. Package006 at 1e-7 is retained unchanged as the third comparison.

At 1e-6, 33 rows exceed the 1e-6 derivative threshold (maximum 3.67208e-5 at tet1635). At 3e-7, 2 rows exceed it (677 and 1635; maximum 3.30476e-6). At 1e-7, the original one-row exceedance remains (1186;1.21490e-6). All center and endpoint projections satisfy the v3 residual and dynamic-correction checks, and the recorded branch predicates are stable at every declared step. No single declared common step passes every nominal-direction row. Every warning and output is retained.

This CPU-only sweep documents step dependence. It does not select a best step per row, replace the original gate, establish a universal noise floor, or qualify a native solve. The separately sealed realized-direction and MP90 analyses explain the 1186 representation/cancellation sensitivity; they do not modify these results.

Run `python3 sweep_uniform.py 1e-6` and `python3 sweep_uniform.py 3e-7` in a fresh package directory; existing output directories reject overwrite. Run `python3 verify_and_seal.py` to verify row domains, projection gates, endpoint sizes, reported derivative errors and immutable input hashes.
