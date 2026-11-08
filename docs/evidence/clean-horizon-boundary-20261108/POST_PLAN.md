# Repaired horizon-boundary native check

Without changing the fixture, policy, seed or timestep, repeat 255/256/257
steps with submission chunks 1 and 8 and one and two environments. Require
zero physical failures, exact timeout counts 0/N/N, and clean rates 1/1/0.
The 256-step timeout must count as a clean requested horizon, while the same
timeout before a 257-step requested horizon must not. Preserve all output and
input identities. This exercises chunk and environment indexing; it does not
by itself qualify policy skill, physics, other termination reasons or all
scheduler modes. The simultaneous CPU reference excludes performance claims.
