# Exact face-active skin source rows in the native importer

The native role consumer now accepts the exact face-active subset of a larger pinned O/I/D source namespace. Compact geometry rows retain their original, noncontiguous source vertex IDs. Each source ID must be unique, in range, and identical in the registered surface, source registration and FEM boundary node. Every compact point must be referenced by a face. Face count, source hash, full86 weights, parent registration, donor frame and coordinate checks remain mandatory. Full-domain inputs retain their strict identity ordering.

The existing `NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=ON` CMake option now also reaches `numi-human-native`; previously only the mass-mapping CLI and profile checks received it. This is the existing explicit 500k-node/2M-cell importer profile. Its default remains OFF. It does not change physical acceptance thresholds.

On SSH Numi (Apple M6), the current-main role-consumer check passes the positive sparse-ID case and six compact-source rejection cases, including transactional failure. Existing full-domain and detailed-transfer controls still pass. Both existing legacy/full capacity tests pass with assertions enabled. The selected integration worktree additionally rebuilt the native executable with the full profile and passed seven relevant importer, mass, facet and volume checks.

These are importer and build results. The selected source-only zero-failure skin/fat checkpoint still requires conservative physical mass admission and accepted native motion qualification; this increment does not claim either.
