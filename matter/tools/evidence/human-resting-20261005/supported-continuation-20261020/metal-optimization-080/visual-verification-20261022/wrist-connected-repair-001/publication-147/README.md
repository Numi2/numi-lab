# Publication 147: offline geometry construction checkpoint

This checkpoint adds the selected 2462 surface-crossing result to the previously published 2517 result. The counted sequence in the retained evidence is 2586 (P145), 2517 (P146), 2513 (round-006-wide-chain-025), 2469 (round-005-barycentric/resume-backtrack-001), then 2462 in round-005-saved-scale-002/post-backtrack-check. The earlier candidates remain in the package with their own source identities and complete reports; the selected candidate is SHA-256 211e0e6c422761633ea5e81c5eaf8ee91797a4cf9c5444bf37e4c6bd1f126b38.

## How the selected checkpoint was produced

The existing quality-projection helper project_patch_quality545.py was used unchanged (source SHA-256 ac230cf06743aab449e9cd34d96b06ece27c35f6c36cbefdd0129311cffdb7c2). The 2x proposal report explicitly records that it is a reused proposal and not a new LP solution. The initial projected candidate passed the recorded arithmetic gate. Local witness backtracking retained each pass: pass 1 had 641 self-pairs; pass 2's area-invalid result is retained; a later pass had 11 self-pairs; and pass 4 reached zero self-pairs but remained CONSTRUCTION_PROGRESS_ONLY. Its depth-merit report says NO_CONSTRUCTION_DEPTH_PROGRESS, so that status mismatch is preserved rather than treated as a clean pass.

A fresh final arithmetic check for that backtracked candidate failed the edge-gradient gate. The retained post-backtrack patch then went through a two-iteration quality projection, after which a fresh complete exact-construction check reported 2462 surface-to-bone crossing pairs, zero contained-node/bank pairs, zero self-intersection pairs, and all 185 banks checked. The exact report records 237 typed anchors, a maximum source displacement of 11.9944701235 mm under the 12 mm cap, a maximum step of 0.7843535285 mm, the component edge-gradient cap of 0.5, and passing full-F32 area gates (minimum current oriented ratio 0.1697134; minimum fixed-reference magnitude ratio 0.10009357). This is a construction result; a reduced crossing count is not a closed or physically qualified skin.

The earlier round-004-barycentric fit is included as a failed attempt. Its launch log ends with a face-target report candidate/target-domain mismatch; it is not a passing construction. The previous barycentric proposal and compatible resumed-seed receipt are also preserved. That seed receipt certifies rebuilt construction seeds and zero-step feasibility only; the separate 2469 full check is the geometric result.

## Scope and limits

All of these are offline construction checks. native_admission is false and accepted_physical_steps is zero. This checkpoint does not claim native execution, physical skin/FEM qualification, a closed skin, visual inspection, anatomy completion, or clinical/biological validation. The 2462 surface crossing count remains nonzero. The point/bank and self-intersection zero counts are limited to the checked candidate and the report's exact predicates.

The package retains failures, partial results, proposals, source snapshots, launch commands, logs, and complete reports. publication-manifest.json pins every included file by size and SHA-256. The selected candidate and exact report paths are listed in that manifest.
