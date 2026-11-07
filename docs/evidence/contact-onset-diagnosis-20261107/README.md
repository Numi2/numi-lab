# Native diagnostic evidence

All source/build/native measurements ran on ssh numi, Apple M6.
The adjacent reports state passed and failed gates separately. This is not a performance qualification.

retained-evidence.tar.gz contains raw attempts, plans, patches, traces/captures and build logs, including failures. archive-manifest.json gives SHA256 of every member; the archive was independently decompressed/read and every member hash verified. Extract into an empty directory. Capture manifests also contain per-array and payload hashes; the FP64 reader verifies both and falls back to the adjacent payload basename when the original absolute remote path is unavailable.

Use the repository tests/contact/diagnose.py and trace_diagnose.py with Python/NumPy2.5.4 for FP64 local equation checks. Use tests/ccd/world/trajectory.py on world-steps.json and the decompressed world-steps.stderr for segment enclosures. The contact refinement tool returns1 on the retained failed force reference. The CCD analysis explicitly retains all_impact_resolution_gate_passed=false.

Per-attempt provenance records977eb61 plus tracked source.patch and native binary/shader hashes. New offline instruments are retained with final source hashes. See DATA_QUALITY.md in the archive for invalid/superseded attempts and diagnostic timing limits.
