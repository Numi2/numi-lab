The unprofiled051 .01-control first-root wall time was slower than050
(16.657867917 vs15.103505375s), despite exact first-row parity. This does not
support a speedup claim. GPU inference: interleaved nine-column scheduling
places column0 in every SIMD group, so the stateless stress-once branch may
serialize both tangent paths within every group. The next bounded trial
will group elements by basis column (same arithmetic/cache addresses/all-nine
publication) and recheck serial/parallel cache bitwise parity plus native
control. This is an inference until timings confirm it.
