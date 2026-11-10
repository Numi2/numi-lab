# Retire verified duplicate trace files

The listed files are raw build-directory duplicates of archived compressed traces. Fetch the exact pushed commit into the independent restoration checkout, decompress each remote Git object and compare its complete SHA-256 with the corresponding raw file. Delete the explicitly listed raw duplicates only if every comparison succeeds, and record hashes, byte counts and remote revision. Compressed Git archives and unrelated data remain in place.
