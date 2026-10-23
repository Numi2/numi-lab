# D35 knee100 source-rebind publication closure

This bundle records the exact D35 candidate code delta and the evidence needed to review or replay it. It is an **unadmitted diagnostic**, not a physical qualification.

## Candidate and exact source delta

Candidate ID: `wholebody-attachment-source-rebind-035-knee100-regenerated-deltas-001`
Candidate identity SHA-256: `aa0670d108f46c391cd67ee9f38200baeeb25ed60a7924cc081c0e782ba4c212`
Candidate100 payload SHA-256: `27368e4bb9a0f59f41e67241a61b17de515a6ffb651b36dce6d47a81902869f3` (26,091,000 bytes; already tracked on main at `matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bilateral-knee-connected-weight-repair-100/candidate.nhskin`, so do not copy it again).

The publication base is `a4584e60fd7ffac10ac10c3c689a1a3ca6c87a9e`. The proposed six-file set is exactly the four app entries below plus this closure's `README.md` and `D35-publication-manifest.json`; hashes and base hashes are in the manifest. Stage the four app entries from `source-overlay/apps/`:

- Add `apps/NumiHumanFinal100CandidateRegistration.hpp`.
- Replace generated `apps/NumiHumanWholeBodySkinReduction.hpp` with the 100-specific source identity, donor frames, weights, and normals.
- Update payload, registered-CSV, and reduction-header hashes in `apps/NumiHumanWholeBodySkinTransferAncestry.hpp`.
- Update `apps/numilab_human_myosim_visual_probe.mm` to include the 100 candidate registration and bind the diagnostic gate to this exact candidate identity.

The D35 tested build used base commit `de561dc67753a67f4b5c0c607bc18af3e42bce01`, an ancestor of the publication base. The tracked `apps`, `include`, and `matter/include` trees are unchanged between those commits. The build dependency file resolves `NumiHumanCoarseTissuePreparation.hpp` from `source-overlay/apps`; that byte sequence is SHA `0f185acc77dc67a49587ba2a20b7873fb6cfba14cdfc18cff30198d579a18c62`, exactly the `a4584e60` header. Do not stage the separate `source/apps/NumiHumanCoarseTissuePreparation.hpp` copy, which was not compiled. The other 18 overlay files match the published base.

The new tables keep 2,411 coarse rows, 4,777 faces, 28,632 parent-refined tets, O/I/D coordinates, topology, thicknesses, and total tissue mass fixed. They change 435 full-weight rows, 60 dominant donors, 434 source-normal rows, and 2,410 candidate C rows. 14 inferred attachment sites pass the existing selection policy; 6 stale inferred hand sites remain omitted and are not replaced. This changes source-donor allocation; it is not an unchanged-mechanics claim.

## What was tested

The Werror compile and link passed on the D35 build snapshot. Exact source, command, dependency, compile, link, and binary hashes are retained in `D35-build-manifest.json` and `build-main-de561-002/`. The source-only initialization reached the intentional unadmitted stop after preparing the tangent source (exit 1 by design); its frozen invocation and raw receipt are under `native-source-only-035-main-de561-002/`.

The native diagnostic accepted 8 steps of 125 microseconds (1 ms total) with registered ligaments enabled. The retained postrun audit reports min/max FEM J `0.965382332239/1.23486808769`, maximum node displacement `0.78739 mm`, 34 registered/passive ligament force rows, contact force residual `0 N`, and contact moment residual `1.86342949465e-09 Nm`. The D34 comparison had the same 157-body reference mass table and peak penetration `0 m` in both runs. These numbers describe only this short diagnostic.

Do not describe the 1 ms result as ROM, natural-skin, anatomical, biological, clinical, or production qualification. The saved `resting-surface-audit.csv` contains only its header, so this run does not establish a full-surface collision audit. The body-mass reference is an inferred global budget, not a measured subject mass. See `native-diagnostic-035-1ms-008x125us-001/postrun-audit.json` for all retained gates and explicit limits.

## Replay and evidence pointers

The candidate table generator and its exact dependency list are `provenance/build_candidate_rebind.py` and `provenance/build_candidate_rebind_inputs.json`. The input config pins absolute evidence paths and hashes; `D35-manifest.json` binds all inputs and outputs. Regenerate only into a fresh output directory, and keep the existing candidate/run directories immutable.

For native build/run replay, use the exact source delta above, the build command/cwd and link command under `build-main-de561-002/`, then the frozen launch snapshots and raw receipts under the source-only and 1 ms run folders. Re-run the post-run parser with `python3 provenance/audit_d35_native_diagnostic.py` from the original evidence layout; that parser also reads the paired D34 run named by the report. The 1 ms launch and parser are bound by the hashes in the publication manifest.

The six proposed publication paths are enumerated in `publication_set_files`; the four app files have exact staged hashes in `publication_set_file_hashes` and the README hash is recorded there. The manifest SHA-256 is reported externally to avoid a recursive self-hash. The builder/config, D35 input/output manifest, build manifest, native launch snapshots, raw receipts, and postrun audit are retained at their original candidate evidence paths and are not copied by this closure; their paths are enumerated in `supporting_evidence_status.retained_paths`. The 26 MB candidate payload, native `.mrvpack`, registered-skin capture CSVs, object files, and executable are also not copied. The unused incomplete-snapshot compile attempt is preserved separately in `compatibility-attempt-001/` and is not a code compile result.
