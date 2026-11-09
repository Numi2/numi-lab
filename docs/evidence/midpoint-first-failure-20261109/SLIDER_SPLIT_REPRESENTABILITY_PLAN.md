# Retrospective split-velocity representability check

For every accepted step in the four retained reacting-slider prefixes,
independently evaluate the full-step rigid momentum equation with the recorded
start velocity, timestep, body mass, gravity and measured body-wrench impulse.
Round the ideal endpoint first to one IEEE binary32 word, then represent its
remaining correction in a second binary32 word. Compare each reconstructed
momentum residual with the unchanged per-step `max(1e-12 N s, 1e-5 * |FEM
impulse|)` gate, using high-precision decimal arithmetic over the exact stored
binary inputs. Retain source trace hashes, maximum residuals and gate counts.

This is only a representability diagnostic with the historical force samples
held fixed. It does not run a midpoint solve, establish the accuracy of the
recorded force, account for feedback from a changed state, test Metal
arithmetic, or replace the required accepted two-way transaction and replay.
