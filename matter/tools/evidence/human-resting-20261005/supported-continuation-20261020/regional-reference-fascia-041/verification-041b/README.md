# Verification 041b

The unified verification completed successfully (exit 0; 37.036548999953084 s). The native invocation advanced two 125 µs steps (0.25 ms total) and passed the injected late-reject/retry check. This is a short mechanics-only transaction test, not sustained support or physiological qualification.

The retained regional fascia evidence shows all 26 source regions admitted, zero unused source regions, and 12 prepared and compiled reference-mode objects (1,202 nodes and 2,127 tetrahedra). `native/selected_stdout.log` preserves every `source_prepared` ownership row, each object’s exact compiled state/element offsets, and projection/admission records with original stdout line numbers. The mass line reports 97.1319507 kg initial and 97.1319508 kg executed rigid-plus-FEM mass, including 14.4160078 kg FEM. The late-reject/retry and terminal FEM audits are retained there as well. Independent mask/geometry, contact, mass, and material fixture logs are retained as small files beside the run records.

`source-reconstruction-041b.json` verifies the retained 041b patch against base revision 5c83a94632ad5c7061e8d465cc2271f96f53429d using a temporary minimal Git archive: patch check/apply passed, and all 24 source hashes match the run provenance. The temporary archive was removed. Executable, metallib, and suite-library hashes are recorded in the reconstruction and native provenance files; none are copied here. Raw stdout and large native output artifacts are not copied; their original paths, byte sizes, and SHA-256 identities are retained in `native/raw-log-identities.json`.

The failed initial 041 preparation attempt and its patch remain preserved at the parent evidence directory. This passing 041b package does not replace or erase that failure record.
