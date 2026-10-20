# Generation-safe private predictor mode
After the frozen 043 test completed 32/32 steps (4 ms, 680.7863612920046 s wall), review identified that the temporary reserved-word marker could coincide with a valid physical generation. Require an additional private high flag bit in the line-search-only microstep copy. Ordinary physical flags never set that bit. No physical state, equation, clock, tolerance or acceptance limit changes.

The completed 32-step result remains bound to source-043.patch. Validate this publication guard with a rebuilt vascular owner fixture and a two-step integrated late-reject/retry. Require the new accepted CSV to be byte-identical to the tested predictor prefix. Do not relabel the 32-step binary as the guarded binary.
