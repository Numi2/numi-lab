# Focused surface-sort checker baseline

This evidence package binds the focused checker source and target build to the retained legacy shader library. The current library is specifically the `cardboard_contact_geometry_check.metallib` produced by the focused target; it is not the main NumiMatter library. The retained legacy library and both exact raw key/index/count receipts are identified by repository-relative paths and independent SHA-256 hashes in `manifest.json`.

The legacy and current sort receipts are byte-identical. The package also preserves the original 112-row geometry-check output in `geometry-112.log`. Cases cover two environments, empty and all-inactive inputs, odd active/mixed sets with equal-key ties, cache skip/bypass controls, and 20,000 primitives using the static FEM surface-map count.

The test is a synthetic production-kernel ordering fixture. Its timings are diagnostic only and make no physical or whole-scene performance claim. `checker_source.mm` is the exact source snapshot used for this build.
