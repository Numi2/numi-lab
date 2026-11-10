# Restore and retire exact duplicate navigation traces

Independently fetch the expected published branch revision into the existing restoration clone, stream and decompress each of the 36 named archive entries, and compare its full SHA-256 to the exact raw local file. Only after every comparison succeeds, remove those named raw duplicates and record their hashes and byte counts. Leave all published evidence, unrelated directories and active run files untouched.
