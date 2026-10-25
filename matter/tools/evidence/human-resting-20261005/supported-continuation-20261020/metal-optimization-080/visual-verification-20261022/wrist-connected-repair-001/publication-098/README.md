# Publication 098 - local-witness backtracking evidence

This staged increment is based on published P097 (7b5b01d6e70de19c5c6bceb5a370d4117bcc3bc8, tree 2d52a213d9e77b3a876ff826b143a41ad87b0032). It updates the connected-surface fitting owner with a tested local witness backtracking path and retains the measured 185/186 construction attempts. The source checkout and index were not changed; this directory is staged evidence and is not committed or pushed.

## Source and tests

The canonical owner is canonical/matter/tools/human_skin_fat_surface_fit_constraints.py (SHA-256 05b1beb2105a184deb885974db892fa0f8e519fc3e1fdfde5aa441c180c97b34). The new owner regression is canonical/matter/tools/test_human_skin_fat_local_witness_backtracking_183.py (SHA-256 686acc7c91e397aa3d4ffcc6423ef87cf4e1209556541faf91e71da385a2628d). The previous owner bytes are retained in evidence/original-owner-20526.py (SHA-256 9175186e4a916dbfd8b2ce4d10f4bd7bb09938cecf442b291bc093823a1336e7). Unchanged tests and their NPZ/Python fixtures are copied to the canonical test directory so this staged suite runs independently.

Fresh SSH Numi verification used:
  /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q -p no:cacheprovider test_human_skin_fat_surface_fit_constraints.py test_human_skin_batched_node_projection.py test_human_skin_fat_local_witness_backtracking_183.py

Result: 56 passed, 10 subtests passed (evidence/owner-tests/fresh-run-002.log). The initial staged run is also retained as fresh-run-001-missing-fixture.log; its three failures all came from the not-yet-copied NPZ fixture, not from assertions. The copied fixture hash is d93fb71ba2ee18bf7743e20c7932ff47aa6f58e27db284de8e2060653ea58494.

The local-witness helper's as-run and final snapshots, tests, and dedicated test log are preserved under evidence/183; these are provenance for the owner change, not geometry qualification.

## Construction evidence

The run starts from the published P097 184 pass-006 candidate (SHA-256 f0e3129826db621399cc03f889baf7a43abc1526b17f338562c84f062f9b398a). The 185 bounded local continuation did not accept a step. The 186 pass-030 output is retained as the selected construction-progress candidate, SHA-256 5cf00286e56475740ae965c7e6b2d58b81357fc3d260a3e4d90cc879ac14cb77.

On the matched 18,430-point / 36,815-face domain, the complete typed-bank point audit decreased inside point-bank pairs from 1,520 to 1,174 and maximum reported depth from 5.256748 mm to 4.756869 mm. The all-face oriented-area floor was 0.10253 and fixed-reference area-magnitude floor was 0.14443; anchors remained bitwise unchanged. The complete typed-bank surface query covered all 185 banks and found 6,161 candidate-to-bone triangle crossing pairs across 30 banks. Separately, the exact PLC self-surface audit found zero candidate self-pairs.

This is construction progress only. The candidate introduces two typed point-bank pairs with depths 2.649991 micrometers and 0.079445 micrometers, and the complete audit does not certify edge or solid clearance. It is not admitted, has zero accepted physical steps, and is not evidence of native runtime, mass, or physiological qualification. Later attempts 187-189 are intentionally outside this publication snapshot.

The 185 gate-correction receipt explains that an older fixed-original signed-normal test rejected 16 faces in the already valid starting geometry. The corrected check compares orientation against the current/start surface while retaining a separate fixed-reference area-magnitude floor; this is a frame-consistent construction gate correction, not a relaxation of native physics gates. Failed 185/186 attempts and candidate hash inventory remain under evidence.
