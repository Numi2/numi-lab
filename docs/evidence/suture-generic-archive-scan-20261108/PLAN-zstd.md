# Additional zstd member-name scan

After the primary scan, inspect the three remaining zstd archive blobs from the same frozen inventory by streaming Git blob output through the installed zstd decompressor into a tar member-name reader. Do not extract. Apply the same member limits and name pattern, record both subprocess outcomes and a separately streamed compressed-byte hash. A contained Git pack or other nested file is not internally searched by a tar name listing. The result is still only a name scan, not checkpoint discovery or replay.
