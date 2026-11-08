# Standalone host-state replay

In the predeclared 128-node, two-offset, 41-step fixture, the accepted state after step 20 was written to disk and read into a staged state. All 20 subsequent accepted updates matched the continuous run bit for bit for positions, velocities, twists, twist rates and both attachment reactions in both environments. A truncated input was rejected without modifying the existing destination. The exact checkpoint remains on the host and its hash is recorded.

This checks the test fixture's explicit host-state serialization and standalone continuation. It is not a production checkpoint format, arbitrary-corruption detector, a cross-process test, resident MetalWorld import, contact-history restoration, coupled rollback, or replay of the historical surgical state. The owner/shader are pre-existing native binaries and independent clean rebuild remains open.
