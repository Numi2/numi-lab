# Integration branch verification

At integration source `6e960159aeaf809bedf931498417f5179c7de59d`, the native integration build passed `metal_world_small_impulse_residual`, `wave32_status_reduction`, and `metalrobo_metal_world_contact_probe` on Apple M6. `integration-ctest.log` and `integration-contact.log` are retained; their incidental timings are not performance results. The source combines the main reporting fix with the newer integration branch and does not promote either disabled experimental solver option. These checks do not establish mixed-contact convergence, calibrated forces, or whole-owner rollback.

`integration-inputs.sha256` binds the executed library, shader, and probes on `ssh numi`; binaries are not committed. `INTEGRATION-SHA256SUMS` covers these supplemental evidence files without changing the original evidence bundle.
