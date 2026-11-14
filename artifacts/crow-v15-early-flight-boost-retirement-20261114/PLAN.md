# Verified trace retirement

Both short compressed traces were independently restored and decompressed from the pushed GitHub commit with matching sizes and SHA-256 hashes. Preflight only those two local and two remote duplicate traces for hash, inode, link count and open handles, publish the preflight, and recheck before retirement. Replay, source, analysis and remote Git copies are preserved.
