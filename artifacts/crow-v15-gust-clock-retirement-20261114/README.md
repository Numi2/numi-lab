# Verified duplicate retirement

The gust-clock raw and split stream were independently retrieved from GitHub and checked against the recorded concatenated and per-part hashes and gzip EOF in the observer artifact. Preflight checks regular file, link count, physical allocation, non-dataless status, SHA-256, open handles, inode, size and modification time. The local part002 was flagged dataless and is deliberately excluded and retained. Only the exact files in the pushed preflights are eligible for retirement; checks are repeated before unlinking and receipts are archived.
