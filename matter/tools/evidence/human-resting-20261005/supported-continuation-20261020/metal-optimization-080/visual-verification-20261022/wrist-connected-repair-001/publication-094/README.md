# Publication 094 — connected clearance evidence prefix

This bundle freezes completed fit 006, failed fit 007, coherent face-first target runs 001–003, the current-normal diagnostic, the deferred-before-fit receipt for 008, and the isolated barrier-helper numerical repair. It is based on published origin/main commit 37eabf9ab930ff7911964ada5aeac1831cfc760c (tree 96c4fcb95dd023fd5fe22b29fefd45b25684ca59). P093 dependencies are referenced by their immutable Git paths and hashes in verified-published-inputs.json; old source and bank exports are not duplicated.

Fit 006 accepted one SCP construction step and then stopped because new query rows required fresh targets. Its frozen candidate audit reports 3,626 surface-pair crossings and 13 inside point/bank pairs. The run was not a physical step: the evidence records zero accepted physical steps and no native admission.

Fit 007 failed before a new fit step because the rebased self-separation baseline was infeasible by 4.263383146962953e-7 mm against the helper's 2e-8 mm threshold. Its unchanged as-run sources, log, and intermediate files are retained. A separate tested barrier helper from 176 is included with its 6-test log; it is an unintegrated proposal, not a rerun or repair of the frozen 007 output.

Coherent face-first target run 001 contains two loader failures before target work. Run 002 completed with 3,658 matched surface pairs and 3,532 certified source-outward first-union target proposals; 125 source-winding rows remained unresolved, and no geometry fit or admission was performed. Run 003 completed against the fit-006 candidate with 3,626 matched surface pairs and 3,501 certified proposals; 125 source-winding rows remained unresolved. Its as-run report still has output_sha256_pending: true; the completed execution receipt carries the output NPZ and JSONL hashes, and both records are preserved.

The current-normal diagnostic found 421 of 3,501 source-oriented rows opposed to the current deformed-face normal. Therefore the frozen 008 receipt records that fitting was deferred; no 008 driver or geometry output is included. A later current-normal fit (009) is outside this frozen prefix.

All content is construction-target, fit, diagnostic, or helper evidence only. No geometry, mass-transfer, tissue-mechanics, runtime, or physical admission is claimed; accepted physical steps remain zero.
