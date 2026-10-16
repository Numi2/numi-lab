# Portable compensated FP64 reference seal 001

This seal binds the additive R-ready reference files, their patch and CPU tests, the frozen source173 equation base, and run018. Hashes are in manifest.json; the manifest excludes itself from its own hash list. The exact modules used by run018 are separately bound under run018.portable_tools_used.

Seven qualified-Python unit tests passed; py_compile and additive patch dry-run/output comparison passed. Earlier failed test attempts remain included and labeled; they were harness/import-path mistakes, not evidence for the equation gates.

Run018 passed all 1,728 center projections and all 311,040 strict endpoint projections. The 1e-6 derivative adjacent-pair criterion passed for 15,536/15,552 row-directions. Sixteen remain unresolved: 4 with no passing grid point and 12 with only one. Their full eps-by-eps records remain in the bound run018 artifact. Nothing is converted to pass by this seal. No absolute noise floor was measured.

This is CPU-only reference evidence, not native Metal or physical qualification.
