# Retire verified duplicate experiment executables

For seven completed experiments, independently fetch their exact committed runtime archives from GitHub, compare each archive and every contained probe and diagnostic library with the local copies and recorded SHA-256 and byte count, then remove only those 14 verified duplicate build files. Keep sources, archives, shared dependencies, active runs, logs and all unrelated work. Abort without deleting anything if any comparison fails.
