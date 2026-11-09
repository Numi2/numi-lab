# Midpoint stage precision: short native transaction

On the opt-in research path, n=8 at 3.90625 us accepted and replayed all 205
requested steps, including the previously failing step 204. Forced rejection
restored physical state exactly and the continuation replay matched. The
minimum gap was 0.000196854656679 m, so this was a noncontact segment. The
maximum independently evaluated per-node residual divided by node gravity
impulse, without a fixed denominator floor, was 1.81755e-5 (limit 0.005); the
maximum kinematic defect was 1.86457e-20 m. The scope and malformed-snapshot
regression also passed. This is not contact-force, spatial, dynamic-owner or
performance qualification. The 10 ms contact-bearing attempt is separate.
