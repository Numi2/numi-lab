# Retire verified raw flight trace

Fetch the pushed research branch independently, decompress the archived flight trace from that remote commit, compare its SHA-256 to the raw task-owned trace, then delete only the verified raw duplicate. Preserve the compressed Git archive and record the commit and hash.
