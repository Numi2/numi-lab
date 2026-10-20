# Geometry reference expansion 039

This package records a standalone checker sequence. The first host run failed with the geometry-oracle mismatch recorded in initial-run.log; its cause remains unresolved. The exact source and executable for that first failed run were not independently retained.

The later diagnostic run, deterministic repeat, explicit-flags run, and two post-failure comparison builds (original and strict compiler flags) passed. The initial and diagnostic 130-row GPU payloads are byte-identical. The final explicit-flags run reports 130 rows and two direct exact VF JVP cases; the check reports zero physical transactions.

This is standalone geometry and arithmetic evidence only. The repeated and recreated passes do not explain the first failure, and no integrated simulation or physiology claim follows from this package. The source-binary-payload-manifest.json binds current source/dependency, build artifact, run payload, and comparison binary hashes. The comparison source is retained only as a post-failure reproduction input, not as the exact initial failing source.
