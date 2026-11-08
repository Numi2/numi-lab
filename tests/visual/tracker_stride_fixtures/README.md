These deterministic zero-policy contracts were generated before correcting
object-track output stride. `0.npolicy` uses compact actor history with no
visual suffix; `1.npolicy` includes an extended masked-depth suffix. The
fingerprint test requires the compact contract to remain compatible, the old
extended-stride contract to be rejected, and an explicitly rebound contract to
compile. These fixtures are not trained policies.
