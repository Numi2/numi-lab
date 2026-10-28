# Existing TetGen to native mass-mapping import 218

The source adds a strict offline reader for existing TetGen `.node`/`.ele` files and separately recorded child material/geometric-donor/anatomical-region/stable-source-face labels. It uses the existing physical child bundle and conservative mass transport. It does not add a per-step CPU implementation or another volume artifact framework.

Import rounds positions to Float32, rechecks positive volume, bounds counts (300,000 nodes, 200,000 cells), preserves explicit node identities, verifies element/material label agreement, and rejects duplicate cells, duplicate node IDs, missing nodes, nonfinite coordinates, truncated/extra rows and invalid donor domains. Import is transactional. The mode is explicitly reference preparation with current equal to reference; it does not claim a moving native frame or physical volume admission.

Strict tests passed with the production Apple compiler on SSH Numi. The reference control imported the actual native parent's 7,233 nodes and 28,632 cells, then exercised the existing mass mapper. Its final build took 4.171350958 seconds total / 3.892886125 seconds mapping; all 28,632 dominant parents matched. Total parent mass was 13.939678963460464 kg; child mass differed by 3.5527136788005009e-15 kg. All 148 material/source-donor budgets closed within 3.9968028886505635e-15 kg. The exact donor/region mismatch count (1,532 cells) is retained. Original parent packet and labels are in checkpoint 217; import inputs and replay command are included here.

The initial control before the duplicate-cell gate is retained separately from the final build control. Their file/binary hashes are recorded. Fixtures and the unchanged-parent control establish parsing and mass-transfer correctness only.

**Pending:** repaired O/I/D joint gates, positive replacement cells, changed-volume labeling and mass mapping, physical scene installation, and accepted motion. The new zero-crossing skin is not admitted by this checkpoint, and no physical steps are claimed.
