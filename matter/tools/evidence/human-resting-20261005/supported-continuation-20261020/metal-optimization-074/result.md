# 1,024-lane trial result

The M6 Metal pipeline accepted a 1,024-thread orthogonalization group and completed four 0.125 ms mechanics steps with FEM/fascia terminal audits passing and no root assistance. It did not improve performance: body RTF was 1.09076e-5× versus 1.11377e-5× for the matched 512-lane replay (2.1% less throughput), and process wall was 57.65 s versus 56.64 s (1.8% slower).

The endpoint comparison failed the pre-existing 1% numerical-consistency gate: peak deformation max_J was 1.1293475 versus 1.1162471 (1.17%). The trial is retained as a rejected result; the shipped path remains 512 lanes. invocation.json records the source/binary/input hashes and exact environment; stdout.log, stderr.log, summary.json, and analysis.json preserve the measurement.
