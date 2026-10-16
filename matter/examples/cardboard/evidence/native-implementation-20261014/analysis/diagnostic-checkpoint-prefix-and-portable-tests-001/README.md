# Pending diagnostics at an accepted coupon checkpoint

This incremental overlay applies after source161-002. A requested operator-FD
or selector-5 consumer capture may have no files only when the native process
exits successfully after an explicitly requested accepted coupon checkpoint
strictly before that capture's numeric control. The result/prefix schemas,
accepted cursor, both-environment metadata flag and checkpoint/metadata hashes
must agree. Paths remain fresh, bounded and local to the current case.

The resulting status is `validated_with_pending_checkpoint_targets`, with an
explicit per-target reason and no positive qualification. Full normal runs,
failed native processes and already-visited capture targets still reject absent
files. Existing capture consumers continue to require actual validated files.
This does not qualify native restore or uninterrupted/resumed trajectory parity.

The environment and resolution tests now use their checkout's source, with an
explicit test-only source-root option for isolated overlay validation. They do
not require this machine's historical snapshot/overlay directories.

The first seven-test attempt's duplicate diagnostic-directory fixture error and
its exact executed candidate remain in validation-attempt001. The fixture was
repaired; production directory freshness rules were not relaxed. Validation uses
the exact source157 selected files plus the source160-002/source161-002 patches,
then this patch. No native build or GPU work is performed here.
