# Fresh synthetic-skin robot approach

This is a new trajectory from the authored initial scene, not the previously archived step-1870 first bite or the dual-PSM receiver state. Source commit `9cc78d2280e6526017e253bec392d0c1052fc1d1`; Release build said no work. The exact copied executable SHA-256 was `912dc9264a79881ccf2b5305ad11b10c9c8073f95efd319ab206a70c8fd6709f`, dylib `83974ddcb828d44c6beb559a47032489ff254100f11b3b83dac063a36a98f45d`. The two shaders and authored synthetic material came from compile-time absolute paths; their hashes were checked before and after, and copies are in the archive. Manifest SHA-256: `16da3656026cd869429cd53320ea7fcbc1efd91c0ae2aa6120400bb30e6a4df4`.

On 2026-11-07 23:54:02–23:54:28 UTC the native `--tissue-robot-first-bite-drive-only --synthetic-skin` run exited 0 with all inputs unchanged. The authored-reset closed gripper settled, then 12 effort-driven groups advanced 12 ms; actual needle-tip advance 25.863 µm, final jaw contacts 6/7, 46,080 active tetrahedra, zero puncture channels, zero removed mass, minimum determinant 0.999999821 and zero failed steps. The authored skin parameters are not a physical calibration; reset was already grasped. This is a bounded free-space approach, not physical pickup, puncture, or a stitch.

The 10 MB archive includes executable, dylib, shader/material copies, logs and the independently hashed grip and approach TSV/Matter pairs. Archive SHA-256: `2d39c80705d0b15997dbfe77f2bd259c245a3289aca55ae76321bee5e3d2c426`. An independent extraction verified all four output files against their original manifest. The new approach TSV SHA-256 is `aae16c75c7a39eb00643a386b601e176cf5faf05e3efe6b93ca532188433bec2`, its Matter sidecar `a6448627449ceeb932bfd54ef435de9aed0848db365bfc40bc3361a03bf97afd`. Byte hashes alone do not establish a successful native restore; that remains the next gate.

## Earned continuation 001

The archived approach at state step 237 restored using its exact pair with `authority_byte_exact=yes`; no physics was advanced by restore. A separate full cooked Newton-budget continuation then used that same saved input and completed 12 groups / 384 base DER substeps (24 ms), exit 0 at 2026-11-07 23:56:58–23:57:17 UTC. Actual needle-tip advance was 111.193 µm, tip-to-skin-node separation 485.098→376.533 µm, jaw contacts 6/6, 46,080 active tetrahedra, no channels or removed mass, positive minimum determinant, and zero failed steps. Runtime and prior checkpoint input hashes passed again after the run. Maximum recorded MetalWorld normal residual was 0.00669797 m/s; this observation is retained and is not a force-accuracy qualification.

The new output TSV SHA-256 is `5a5f9dd59ceb6ffd1c6ed6250ae47515512a73c6ddaac1718f420dd4e30fa1a9`, Matter sidecar `4f963497a25d9ad8c92941dc64637b40d55132cc1bd697da6ae5ff3804e14993`. `continuation-001.tar.gz` SHA-256 `ad83da609a093e14a466837c16931bbba8fdd73bce75dc5ecc272fade3242549` holds the logs, receipts, hash checks and both output files; independent extraction verified them. This output was subsequently restored in continuation 002. Neither this continuation nor the authored-skin material establishes biological realism or a stitch.

## Earned continuation 002

The preceding output restored at state step 621 with `authority_byte_exact=yes`. A 12-group full-budget continuation completed 2026-11-07 23:58:22–23:58:41 UTC, exit 0, with input hashes unchanged. Needle-tip advance was 113.407 µm, node separation 376.533→263.267 µm, final contacts 7/6, all 46,080 tetrahedra, no puncture or removed mass, positive minimum determinant and zero failed steps. Maximum recorded normal residual was 0.00293202 m/s. This remains synthetic free-space motion; independent work has found a rod axial reaction/work failure, so the strand's physical force is not qualified.

New TSV SHA-256 `6bc9d7b9f22705154b804a5517741e2aa15de26a850606e5cd189b079e6a49b2`, Matter sidecar `91decb3aee2fcba7ae298cddf861fe01cf4c55eb66cc6d4aeb395cd49d1c6602`. `continuation-002.tar.gz` SHA-256 `469e46972a4bcfe5af715916445a02beb5216ae505c7ddc8fcb0b496ab7fd4a2`; an independent extraction verified both outputs. This output was subsequently restored in continuation 003.


## Earned continuation 003

The preceding state 1005 restored with `authority_byte_exact=yes` and no physics advance. A separate 12-group, full cooked Newton-budget continuation ran 2026-11-07 23:59:38–23:59:58 UTC and exited 0. Actual tip advance was 113.903 µm, node separation 263.267→150.459 µm, final jaw contacts 8/6, with 46,080 active tetrahedra, no puncture channels or tissue reaction, no removed mass and zero failed steps. Maximum recorded MetalWorld normal residual was 0.00448667 m/s. Runtime and input hashes were unchanged. These are synthetic observations and do not establish the force accuracy of the surgical strand or validate Metal rod mechanics.

The output TSV SHA-256 is `3976b859083d35f9ae6b0aa325c15872db77e6442cd724959e98053520d0f128` and Matter sidecar `ff98b89b4aad84d24d62818857f09719453fec5d447393832ed630ff826974de`. Archive `continuation-003.tar.gz` SHA-256 `6222befffa45f28d6de6722191d271fc568667b9e759d4b0a9ec2e4af741f269`; independent extraction verified both output files. This output was subsequently restored in the guard-stop attempt below.


## Continuation 004: guarded stop

At 2026-11-08 00:02:18–00:02:19 UTC the pinned executable restored state 1389 from continuation 003 with `authority_byte_exact=yes` and `physics_advanced=no`. The subsequent 12-group full-Newton free-space approach returned exit 1 before publishing a new checkpoint: `full-Newton robot approach can cross the tapered-tip contact guard`. The wrapper returned 6; all input and runtime postchecks passed. This is a guarded rejection, not a completed approach, and no output state was produced. Archive `continuation-004-guard-stop.tar.gz` SHA-256 `d205b6e17b3511fae15107a4d133c1e1f63e73cfe3630199bf7affc9c634079d` was independently extracted and every retained file compared byte-for-byte. The script, logs and receipts remain in the archive.

The current full Metal rod witness independently found a large axial force discrepancy in a separate pinned-edge fixture; its magnitude has not been measured in this trajectory. Neither acceptance here nor a later entry would establish surgical strand force accuracy. The existing entry-only stage is designed for the observed 100–200 µm separation and retains its own physical and solver publication gates; The subsequent run and its rejection are recorded below.


## Entry 001: contact candidate rejected

The existing full-budget `--tissue-robot-first-bite-entry-only` path was run 2026-11-08 00:03:30–00:03:48 UTC from the exact state 1389 after byte-exact restore. Sixteen groups produced a diagnostic candidate with 82.126 µm tip advance, separation 150.459→70.412 µm, one tip contact, accepted tip impulse 1.11096892e-8 N·s, tissue reaction impulse 1.11192191e-8 N·s, and maximum observed tissue displacement 2.0825e-9 m. It reported 6/6 jaw contacts, zero puncture channels, all 46,080 tetrahedra and no failed underlying steps. Maximum recorded normal velocity residual was 0.00383542 m/s. The material is authored synthetic, and the unqualified rod force and solver residual preclude a force or biological accuracy claim.

The native process rejected the final aggregate publication predicate with `resumed robot approach lost its grip, thread, or skin authority` and exit 1; the wrapper returned 6. No checkpoint was published; runtime and input hashes passed afterward. The candidate log is not an accepted physical state, and this run does not independently establish full transactional rollback. Source inspection identifies one definite failed conjunct: starting with zero channels requires `channels == 1`, whereas this candidate reports zero. Other aggregate predicates were not individually instrumented and may also have failed. Do not weaken this gate without determining the intended contact-to-channel contract and independently qualifying it.

`entry-001-rejected.tar.gz` SHA-256 `5863f9844dbaacbba467b8e83684633950c3fe8202d6f6f3f6f3a781a4a6ec3e` contains the pinned script, native logs, receipt and input checks; all extracted files were compared byte-for-byte with their source. Script SHA-256 `2b5f3a789d064241567e8583bae525961ff9c751aa12cde9c9b37153022eb8af`.
