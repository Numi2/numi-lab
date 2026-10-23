# Human FEM-node / moving-bone triangle contact regression

Run on the designated Numi Mac, from any directory:

```sh
python3 tests/numi_human_fem_body_triangle_contact/run.py --build-dir /tmp/numi-human-contact-fresh-run
```

The output directory must be new. The runner builds the Metal kernels with the project's existing Metal 4.0, `-O3`, and `-fno-fast-math` settings; runs the CPU derivative oracle and actual GPU owner force/wrench/audit kernels; and syntax-checks the production host owner with warnings treated as errors. It records commands, source and binary hashes, logs and elapsed time. CPU force evaluation exists only in test support.

The opt-in 64-byte sample carries one prepared body-local triangle and explicit slave tributary area. It preserves the existing plane and elastic attachment branches. The triangle query requires an authored outward winding and valid narrow-phase neighborhood. It is not a mesh broadphase, global inside/outside test, integrated scene attachment or implicit contact solve. Existing one-sample-per-slave admission remains intact. These tests establish local force/tangent behavior, not natural joint range, timestep stability or whole-body skin qualification.
