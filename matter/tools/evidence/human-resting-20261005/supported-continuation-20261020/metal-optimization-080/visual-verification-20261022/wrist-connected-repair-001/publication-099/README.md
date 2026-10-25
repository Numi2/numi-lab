# Publication 099 - connected local continuation evidence

This compact evidence increment is based on published P098, commit cd85ac950100270989d8aa9633795c0f2e9b6fb2, tree aacedcdfa34b4c9c00a15ee015c9698466c7b6b2. The exact P098 manifest, owner, starting candidate, helper and runner are pinned in verified-published-inputs.json. The P098 owner is SHA-256 05b1beb2105a184deb885974db892fa0f8e519fc3e1fdfde5aa441c180c97b34. P099 makes no canonical code changes; the manifest explicitly records canonical_changes as an empty list.

## Runs 187 and 188

Run 187 starts from P098 candidate 5cf00286e56475740ae965c7e6b2d58b81357fc3d260a3e4d90cc879ac14cb77. Its first two completed continuations reduced candidate-to-bone surface triangle crossing pairs from 6,161 to 5,496 and then 5,101; point-bank inside pairs fell from 1,174 to 927 and then 716. The third bounded local backtrack made no combined progress. Its final attempted pass had 11 exact PLC self-surface pairs and was rejected; no physical steps were accepted. The 187 output mesh files are hash-only; the summary and exact audit reports needed to trace these steps are retained.

Run 188 reuses the published P098 local-witness owner and starts from 187 round 2. Its selected candidate is SHA-256 b6f431209e8c5c1d0cd6704baee8463f4f55db1c520747a7eedc79b10b39401b. It reduced candidate-to-bone surface crossings from 5,101 to 4,889, point-bank inside pairs from 716 to 556, and maximum point depth from 3.757101 mm to 3.385128 mm. Its separate exact PLC self-surface audit found zero self-pairs. The complete typed-bank report covers 185 banks but still has 4,889 skin-to-bone triangle crossing pairs; the point audit still has 556 inside pairs. Edge and solid clearance are not certified.

## Completed run 189

The eight-round driver exited after five progress rounds followed by round 6 with NO_COMBINED_PROGRESS. The selected round-5 candidate is SHA-256 7aad7ab7286cc89ae006f90dc35e72d8d60f8178023f4ebbc3a9464fee3a0c21. Across the five progress rounds, typed-bank surface crossing pairs decreased from 4,889 to 4,743, 4,649, 4,518, 4,439 and 4,397; point-bank inside pairs decreased from 556 to 473, 402, 383, 351 and 335. The final self-surface audit found zero exact PLC self-pairs, while the complete typed-bank report still found 4,397 candidate-to-bone triangle crossing pairs. Maximum reported point depth remained 3.385128 mm. No physical steps were accepted.

P099 retains the selected 188 and final 189 candidate meshes only. Intermediate 187/189 meshes are represented by hashes and their result/audit records; 190/193 are excluded. The immutable 187/188/189 runner sources and generic constraint helper/tests are included, while no further test or geometry run was performed for this evidence staging.

All runs are construction-only. None is admitted, none has accepted physical steps, and these records establish no native-runtime, mass, or physiological qualification.
