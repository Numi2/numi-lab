# Search current branch trees for the earned surgical checkpoint

Before the search, freeze the current `git ls-remote --heads origin` inventory. Search each available distinct branch-tip tree, not only branches whose names mention surgery, for tracked paths that combine surgical or needle/stitch terms with checkpoint, state, archive or receipt terms, and explicit receiver-bridge/pull-complete phase names. Include compressed and binary paths; retain raw blob object IDs and all branch associations. Set GIT_NO_LAZY_FETCH=1 and GIT_OPTIONAL_LOCKS=0; record unavailable trees or blobs rather than implicitly downloading them. This is a read-only path inventory: a match is only a lead, and a missing name does not rule out generic archive contents, old commits or untracked objects. Inspect candidate metadata, archive manifests and actual stage only after inventory; do not promote, modify, fabricate or restore any checkpoint. Record hashes and incomplete searches.

## Archive inventory amendment

After the initial path search produced no stage-specific tracked checkpoint, inventory distinct tracked `.tar`, `.tar.gz`, `.tgz`, `.zip`, `.tar.xz`, `.tar.zst` and `.tzst` blobs at the same frozen 119 branch tips, without assuming the archive name contains a surgical term. Classify locally available blobs and sizes using no-lazy-fetch object checks. Do not extract or automatically fetch missing objects. This inventory identifies what can be inspected safely; it does not claim that archived payloads have been searched. Record any unavailable branch tree or object and do not interpret it as a negative result.

## Surgical continuation archive-name inspection

The inventory found 18 archives under `numi/evidence/fresh-robot-approach-20261108`, including generically named continuation and partial-state files. Before reading them, stream their tar member names from the available Git blobs without extracting or changing files, retaining archive hashes and any malformed or unreadable archive. Search for checkpoint/state/receiver/bridge and suture metadata names. These archives pertain to a fresh robot approach; any candidate still needs its actual recorded stage and provenance examined before it can be called the requested tissue-bridge checkpoint. Other archives remain inventoried but uninspected by this step.

## Bounded archive receipt inspection

For the 18 already enumerated fresh-approach archives, read only the named
receipt and pre/postcheck documents up to 64 KiB each, in memory, without
extracting archive files. For the two archives containing the staged full
runtime and partial-owner manifests, inspect only their small launch, source,
hash, receipt and manifest text documents. Record names, sizes, contents and
errors; a stage label in a receipt is a lead, not proof of a later bridge.
The remaining generic archives and older commits are not covered.
