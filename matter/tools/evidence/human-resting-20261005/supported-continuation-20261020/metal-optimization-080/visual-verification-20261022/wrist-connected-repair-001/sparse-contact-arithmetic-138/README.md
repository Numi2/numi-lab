# Sparse contact arithmetic regression

The sample-domain force assembly now preserves the original two-add local force accumulation before adding to incoming FEM forces. This removes contraction-rounding differences found by the large strict-parity coupon. All three alternating GPU pairs (7233 nodes, 2411 selected unique shuffled slaves, 4822 untouched rows) match the dense GPU oracle bit for bit. Median kernel duration is 15.833 us versus 953.583 us; the approximately 60.23x ratio applies only to this force kernel. The original strict-parity failure is preserved alongside the exact result.

The full shader compiles with Metal 4.0/O3/no-fast-math; the paired host runtime is unchanged from135. The full native D36 scene accepts eight125-us steps (1 ms). Four monitored endpoint changes versus the existing matched baseline remain below1%; this is neither full-state parity nor anatomical-motion or sustained-performance qualification. Native body time41.414s gives0.00002415x; no0.1x claim.

No geometry, mass, contact admission, controller state ownership or force equation changes are made here. Connected joint anatomy and broader collision coverage remain under repair.
