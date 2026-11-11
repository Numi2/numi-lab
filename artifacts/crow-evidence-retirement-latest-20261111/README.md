# Verified retirement of two duplicate raw replays

Both compressed replays and their manifests were fetched in an independent clone of the pushed commit. The compressed hashes matched, decompression succeeded, and the restored raw hashes matched the build copies. After checking that neither copy had open handles, only those two redundant raw files were removed; the Git-tracked compressed evidence remains intact.
