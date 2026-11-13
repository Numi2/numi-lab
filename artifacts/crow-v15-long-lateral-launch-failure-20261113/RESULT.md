# Startup contract mismatch, no simulation

The first 36-second launch exited with an incompatible policy/task contract before stepping. The passive model observer was mistakenly built from the intermediate wing-hold body, omitting the obstacle-free scene-mask changes used by the reference run. The policy fingerprints correctly rejected that mismatch. The startup model metadata and complete compressed stderr are retained here; no flight result is claimed. The correction is to apply the observer-only metadata print to the exact obstacle-free c_api source, rebuild, and require the already-preregistered exact trajectory and mask parity before interpreting a run.

Preflight commit: 74f2ca6738d71afb9a2ff6dc9a8fb59e9b1ddcd9.
