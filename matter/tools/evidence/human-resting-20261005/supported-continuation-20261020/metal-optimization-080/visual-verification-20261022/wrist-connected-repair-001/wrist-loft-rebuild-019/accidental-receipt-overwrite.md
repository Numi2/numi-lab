# Receipt overwrite notice: `wrist-bone-audit035.json`

The canonical path
`wrist-loft-rebuild-019/wrist-bone-audit035.json` was inadvertently overwritten while importing `audit_wrist_bones035.py` for helper functions. The helper's unguarded neutral-pose audit ran instead of stopping before its report block.

The file currently at the canonical path is the accidental neutral-only rerun, not the prior 035 receipt. Its SHA-256 is `7143d46423350e89ed3110cffe97b002128ebed09c36fbf484ddfe1afca0d347`. An exact byte-for-byte copy is preserved at `wrist-loft-rebuild-019/wrist-bone-audit035-accidental-neutral-rerun.json`.

The canonical file was untracked (`git ls-files --error-unmatch` returned status 1), so its previous bytes are unavailable from `HEAD` or Git history. No reconstruction or inferred replacement of the prior receipt has been made. Do not cite the current canonical file as the original 035 result.
