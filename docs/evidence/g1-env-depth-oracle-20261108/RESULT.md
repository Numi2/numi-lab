# Synthetic depth-helper sample/hold result

The process ran 2026-11-07 23:25:21.074465–23:25:21.293714 UTC and exited 0; all frozen inputs matched before and after. In the research selection kernel that calls the owner's actual `frameSeed` and `measureDepth` functions, the 32 batch output words matched the separate 16-pixel reference dispatches exactly. All 16 held words matched the prior frame. Changing the fresh environment's frame seed changed all 16 tested pixels; forcing the held environment to sample its changed depth also changed all 16. Logs and inputs are retained.

The result is a synthetic helper comparison with separately dispatched pixel records. It does not establish independently rendered scene frames, per-environment renderer cadence, status-gated image publication, resets, or mixed continuation. The renderer guard remains enabled.
