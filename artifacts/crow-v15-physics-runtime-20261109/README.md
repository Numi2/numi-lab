# Isolated v15 development runtime checkpoint

These compressed, hash-bound native binaries reproduce the experimental contact-gated walking / frozen-flight handoff with the isolated task-time articulated-velocity fix and late-contact observer, as tested in `../crow-v15-contact-slip-20261109/materialized-late-route-plan.md`. They are Apple Silicon/macOS build artifacts, not production components or biological qualification. The source patches, actor bindings, positive and negative trials and scientific limitations are in the adjoining ground-contact and contact-slip directories. The two required actor policy packs are archived separately under `../crow-conditional-landing-native-20261109/`.

`python3 restore_runtime.py NEW_EMPTY_DIRECTORY` restores a standalone executable, metallib and `libmetalrobo.dylib`, verifying every uncompressed SHA-256. Supply the recorded model/actor inputs and set the restored directory as `DYLD_LIBRARY_PATH`; original rollout arguments are in the matching materialized-late-route binding JSON. Never replace the default host library or frozen controller with this checkpoint without separate qualification.

The archived runtime was restored on the same Mac mini and reproduced the first 350 physical-state frames of the archived exposed-layout run exactly; see `restore-check.json`. This is a reproducibility check, not flight or gait qualification.
