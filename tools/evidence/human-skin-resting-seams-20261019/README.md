# Local resting-skin source topology repair

This candidate preserves the registered NHSKIN source vertices, normals, donor bindings, and all 86-weight records. It changes only 20 triangle index records, using 10 orientation-preserving edge flips in seven local seam patches. Each replacement triangle retains the two original face IDs as ancestry in `repair-manifest.json`.

The generated payload is `bodyparts3d-myosim-skinned-shell.nhskin` (SHA-256 `d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04`). It has 61,950 source vertices and 108,987 faces. There are no degenerate faces or internal orientation conflicts; all 115 source boundary edges and their vertex pairs remain unchanged.

A direct source-surface audit using the exact q36=q74=0.10 registered-pose CSV accepted the repaired mesh with zero crossing pairs (`source-surface-audit.json`). The audit is deliberately scoped to the source surface; it does not certify generated skin/fat offset surfaces.

The exact repaired asset then completed one accepted 0.25 ms native construction step in the integrated scene. The run uses the q36=q74=0.10 construction pose, the pinned C292 binary/library/MSL listed in `validation-evidence.json`, and reports `resting_aortic_wall_runtime_contact_verified=1`. The coupled CSV hash is `6e25863a30d2ba6b87d0ab6630a4697b1387b03917c0ac4fc8336d72a87485e2`. This is a one-step integration check, not a sustained whole-body qualification. See `../native-first-step.json` for the primary native receipt.

To regenerate the payload from the pinned original, run `python3 tools/repair_human_skin_resting_seams.py --source ORIGINAL_NHSKIN --output-dir OUTPUT_DIR --no-fixture`. By default it also validates the exact compact candidate fixture and registered-pose CSV. `--no-fixture` skips the optional compact-fixture check while retaining the pinned source hash and topology checks.
