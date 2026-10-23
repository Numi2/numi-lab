# Human FEM-node / moving-bone triangle contact regression

From the repository root, run this against a fresh output directory:

```sh
python3 tests/numi_human_fem_body_triangle_contact/run.py \
  --build-dir /tmp/numi-human-bone-triangle-contact-run-001
```

The entrypoint builds and runs the CPU oracle, syntax-checks the production owner and existing surface-attachment coupon, compiles the Metal coupon and metallib, and runs its production-kernel cases. The receipt includes the source revision, source and binary hashes, commands, logs, and elapsed time. For isolated source-overlay checks, the same entrypoint accepts `--source-overlay PATH --base-root REPOSITORY`.

The optional triangle table is an immutable, body-local prepared bank with a per-sample maximum query distance. The selector searches only that bank and fails closed when the accepted node leaves it. This does not search the full bone mesh each step and does not prove the bank contains the globally nearest face or all anatomically relevant faces. It is a local owner-kernel regression, not a global collision, anatomical-coverage, integrated-scene, or natural-range-of-motion qualification.
