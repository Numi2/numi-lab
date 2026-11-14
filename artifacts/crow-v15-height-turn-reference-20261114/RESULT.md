# Matched stronger-height reference

The 20261126 stronger-height trace was reconstructed directly from Git commit 80845dd1. Every compressed part and the entire 253,098,037-byte gzip matched its archived SHA-256, and gzip termination was valid. It contains 1,800 observed state samples for each lane. Those records include steps after the first physical failures of unchanged lanes 0 and 1; comparisons for those lanes must stop at their reported first-episode boundaries.

The first extraction script incorrectly assumed the stream would stop after those failures and asserted 12,324 samples; its failing log and original script are retained. The corrected extractor preserves all 14,400 observed samples. Later comparisons must use the first-episode limits and must not describe post-failure samples as successful flight.
