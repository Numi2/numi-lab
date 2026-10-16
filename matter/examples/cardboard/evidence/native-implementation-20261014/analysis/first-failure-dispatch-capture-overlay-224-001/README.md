# First global failure after dispatch recorder (source224)

This is a default-off, read-only diagnostic overlay on source219. When the full-contact checker is run with --first-failure-dispatch-root CONTROL, the selected activation request is preDynamics, substep 0, and its transaction-XXXXXX-before.bin input is hashed before the command is submitted. The runtime observes the global Matter status immediately after each real production dispatch. The first nonzero status per environment is copied with the solver-certificate buffer contents and the dispatch ordinal; the sidecar maps that ordinal to the pipeline name and binds the accepted input snapshot path/SHA, source/world/program fingerprints, and device.

The recorder adds one small read-only Metal dispatch and buffer barriers only for that explicit selected request. With the option absent, the dispatch wrapper calls return without adding work. It does not change solver status, contact history, accepted/candidate state, or rollback. It sees only global statuses after dispatches; a private transient failure that never reaches the global status buffer is outside its scope. The certificate fields are the certificate buffer contents at observation time and are not asserted to have been produced by the named dispatch.

Suggested activation capture command shape:

cardboard-full-contact-armijo-check <material.nmatter> --sparse-free-body-parity --diagnostic-state-output <new-output-dir> --first-failure-dispatch-root 1199

The parity fixture creates separate dense/ and sparse/ diagnostic subdirectories. The recorder writes transaction-000000-first-failure.bin/.json inside each selected activation directory. A zero-record sidecar means no nonzero global status was observed after a real dispatch. This build is compile evidence only; no native run or physical qualification is included.
