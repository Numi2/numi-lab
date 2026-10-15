# Source 380 tet 674 corrected-input audit

This independent CPU-only receipt checks the corrected selected-point input for tet 674 against the exact rejected-candidate, accepted-state export, and compiled material package from source 380. It also compares the corresponding source 370 artifacts. No native point kernel or GPU was run, and no production files were changed.

The corrected packed material record is 160 bytes. In the compiled `NMMaterialGPU`, `localNewtonIterations` is at byte 80, `bulk` begins at byte 96, and `validity` begins at byte 144. Material 1 in source 380 has `bulk = [679, 293.1499938964844, 0, 0]`, 28 parameters, 15 states, a local iteration budget of 32, and J bounds `[0.20000000298, 2.5]`. The point-input numeric temperature matches packed `bulk.y` bit for bit. Its free-form `case_count_semantics` string still says “0 K”; that text is stale and must not be used as evidence of the numeric input.

Source 370 and 380 have different package fingerprints. The rejected candidate node buffers are nevertheless byte-identical (141,120 bytes; SHA-256 `1ec2da4c997fac19986f5ad196c01dc653457297ae5da767f891d0bd05ea4b92`), as are their full post-rollback material-state exports (971,640 bytes; SHA-256 `6f1b2031dc933cacec54142e9521e914d30a5f8f2889d6bd589e46878c3bfbbc`). Their compiled geometry, tet, and parameter sections match; only material descriptor words at byte 80 differ for materials 0 and 1, changing the local budget 16 to 32. Tet 674’s accepted 15-state history also matches the selected input after FP32 round-trip.

Recomputing the CPU source transcription from the captured FP32 candidate nodes, packed inverse-rest matrix, and material quaternion reproduces the selected `F_high/F_low` words bit for bit. The widened source-transcribed F determinant is `1.0003024223581578`; the direct candidate geometry determinant is `1.0003024133663518`; the native status diagnostic reports J `1.0003025531768799`. The F determinant difference from native J is `-1.3081872e-7` and the direct geometry difference is `-1.3981053e-7`. These are close but do not confirm the native F words. The native J lies inside the compiled validity bounds.

The previous `replay.json`/`audit.json` field `diagnostic_det_difference = 0.002556720636401888` is incorrect: the replay code subtracts `0.99774569272994995` rather than the native diagnostic J. This receipt gives the corrected arithmetic and binds the old files unchanged. Source 380 still ends with status 5 at Newton 0, before FGMRES or outer Armijo. Its exported diagnostic does not expose the local projection terminal reason, so this audit does not attribute the remaining failure to the constitutive update, its temperature, or the iteration budget.

Run the read-only auditor with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/source380-tet674-corrected-offset-audit-002/audit.py
```

`report.json` contains the verification details and SHA bindings. `manifest.json` binds this receipt’s own files and the source 370/380 inputs. The earlier `source380-first-failure-tet674-001` package is retained unchanged.
