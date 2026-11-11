# Retired duplicate raw replay files

For each file, the compressed replay and its SHA-256 manifest were fetched from GitHub using a separate partial clone at the recorded pushed commit. The compressed hash matched its remote manifest, decompression completed, and the restored raw hash matched the local build replay. After checking for open handles and rehashing both raw files, only those redundant raw build copies were removed. The compressed tracked files and full original experiment records remain in the repository. No other files or checkouts were removed.
