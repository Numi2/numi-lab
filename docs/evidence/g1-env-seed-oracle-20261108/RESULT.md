# Independent seed-helper result

The process ran 2026-11-07 23:24:21.244720–23:24:21.477924 UTC and exited 0; all frozen input hashes matched before and after. For the first fresh record, the renderer's included `frameSeed` produced 500282421 in both batch and isolated dispatches, matching the preregistered CPU reference. For the second held record, the three results all matched at 2565727820. The global-pixel-index negative control also passed. Inputs retained their original per-record instance identities and global pixel indices.

These are separate single-record shader-helper dispatches and a host clock calculation, not independent rendered environments. The run does not demonstrate per-environment sensor noise, sample/hold frame selection, metadata publication, task reset or mixed physical continuation. The current renderer remains batch-sequenced and the mixed-divergence guard stays in place.
