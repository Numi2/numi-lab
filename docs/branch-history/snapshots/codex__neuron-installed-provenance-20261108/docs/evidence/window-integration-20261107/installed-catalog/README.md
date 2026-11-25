# Installed authored catalog

The installed wrapper admits an existing authored scene catalog without first
requiring optional built-in meshes. A stale built-in catalog does not bypass
its version and asset checks. CPU bootstrap tests use stubs; the metadata and
bootstrap tests passed. An empty workspace still reports the missing G1, Franka
and PX4 presentation paths and exits nonzero.

A separately installed and relocated package at source commit `88b9b157` ran
`numi window --check` from an external workspace with an authored synthetic
scene catalog, without an explicit scene argument. The one-environment,
16-step check passed, including missing-scene rejection and restoration. Exact
source revision, installed binary and fixture hashes, full output and timing
are retained. The synthetic tetrahedron is not Unitree G1 geometry. The macOS
console was locked; this does not establish a human-observed viewer interaction,
physical accuracy, guided UI training, or state restoration.
