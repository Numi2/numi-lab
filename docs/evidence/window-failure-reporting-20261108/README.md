# Window source dispatch and failure reporting

On Apple M6, an independent source-runtime project launched the integrated
Window using a synthetic Franka WorldPack and a synthetic visual scene whose
camera parent did not exist in that world. The runtime printed
`head-camera parent body is not present in the compiled world`, yet exited 0.
The original plan, result, and stderr are retained as `baseline-*`. This
failure was discovered after correcting source checkout discovery and the
CMake source path; it is not evidence of a working visual workflow.

The revised runtime shares terminal outcome across the worker and AppKit
threads. Unrecoverable startup or restoration failures return nonzero. In
ordinary runtime-check mode, success also requires the check to actually reach
its completion marker. A rejected selection that restores a confirmed working
configuration remains recoverable. The separate training-cancellation check
has its own completion semantics and is not changed by that requirement.

The opt-in `tests/numi_window_failure_reporting_probe.py` uses a separate
workspace (including spaces), both an invalid camera parent and a valid
`panda_link7` camera parent, the same synthetic Franka world, and the same
visual geometry. The invalid case returned 1 with the expected diagnostic;
the valid case returned 0 after reporting the deliberately rejected selection,
restoration, and runtime-check completion. Both input and runtime hashes
matched across cases. The plan and results bind the actual modified source,
binaries and inputs; the probe preserves stdout and stderr in its output
directory. These are bounded synthetic native checks, not physical accuracy,
manual Window interaction, training, or performance qualification.

The fixture WorldPack and related packs can be restored from
`../clean-sparse-package-20261108/fixture-run-and-mutations.tar.gz`; the visual
scene and mesh are in `../integrated-installed-smoke-20261108/run-and-synthetic-fixture.tar.gz`.
Run the probe with their extracted paths and a configured source build; use
`--help` for its explicit arguments. The complete original outputs remain on
this host in `/Users/numi/window-source-workspace-native-20261108` and
`/Users/numi/window-failure-probe-qualified-20261108`.
