# Compacted skin-attachment candidate check

This evidence-only tool selects and validates inferred attachment candidates against the current compacted source map and native site rules. It does not edit anatomy, weights, payloads, headers, or runtime code.

Run the self-test and current-input check with:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/replacement-mechanical-071-attachment-selection-001/select_validate_compacted_attachments.py --self-test
/Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/replacement-mechanical-071-attachment-selection-001/select_validate_compacted_attachments.py
```

Inputs default to the explicit 068 geometry/skin payload, 069 compacted coarse surface, candidate layers, source and row maps, 061 old-site audit, current native/header/test/CMake sources, bone manifest, and 055 native prism checker. Each input path can be overridden with its matching `--coarse-surface`, `--layers`, `--source-map`, `--row-map`, `--source-geometry`, `--skin-payload`, `--left-patch`, `--right-patch`, `--old-sites`, `--source-header`, `--native-source`, `--native-test`, `--cmake`, `--bone-manifest`, or `--prism-helper` option. `--acl-node-count` supplies the exact ACL contribution to the native budget; without it, the tool uses the conservative ACL=0 lower bound. `--output` selects the JSON report path.

The local site checks are source/reduced/donor identity, source weight >=0.85, incident altitude >=50 um, at most 3 sites per donor, same-donor spacing >=25 mm, a matching positive-mass donor-local node, unique active node indices, and the runtime attachment budget. The candidate rows are inferred registration sites, not measured entheses. The native runtime rule is tied to the frozen site-array length; the separate legacy test's 20-site/7-donor assertions are unwired and are not treated as runtime requirements.

Current 069 inputs produce 14 retained active old sites, 6 retired orphan rows, and 2 provisional replacements (16 planned; 7 donors). No site is accepted: native signed-prism checks report 437 skin and 3,297 fat failures, and the emitted-Float32 fat determinant minimum is 3.8165e-6, below the 1e-5 gate. The expected blocked exit is 2. Positive mass and altitude for provisional rows are diagnostic only while layer gates fail. This is not a header/payload proposal or a physical qualification.

Evidence: `selection-validation.json`, `native-attachment-contract-receipt.json`, `self-test.stdout.log`, and `run.stdout.log`. The current full run exits 2 with status `blocked_invalid_or_unqualified_layer_geometry`; self-test passes.
