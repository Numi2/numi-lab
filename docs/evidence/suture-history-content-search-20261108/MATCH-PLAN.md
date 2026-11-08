# Classify v3 schema matches

For each schema-matched archive member in the completed prefix scan, stream that named member from its source Git blob, cap it at 1 MiB, and retain SHA-256, byte count and only schema, phase, step and referenced Matter filename/hash fields. Compare the actual phase to bridge-start and subsequent continuation phases. Preserve all other state bytes in place without copying or restoring them. A matching phase still requires complete coupled-owner and source provenance verification before native replay. No source archive is modified or extracted to disk.
