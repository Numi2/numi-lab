# Branch-tip surgical checkpoint search

The frozen remote inventory contains 119 branch heads. All 119 tip trees
were available and their tracked paths were searched for surgical, state,
checkpoint and named receiver-bridge terms. No path identifying the earned
v3 tissue-receiver-bridge-start checkpoint was found. The result is a path
search, not proof that the state is absent: old commits, untracked files and
generic archive payloads require separate investigation.

The generic archive inventory found 300 distinct tracked archive blobs, all
available locally. Only the 18 archives under fresh-robot-approach were
listed internally in this pass; all 18 were readable. Their name-matched
members and bounded receipts are retained. The staged runtime archive launch
and output receipts identify an earlier synthetic-skin first-bite approach
and grip. The partial-owner archive contains first-bite entry and advance
manifests that explicitly state complete=0 and restorable=0, with missing
owner state. These are not the requested earned bridge checkpoint; they were
not restored or substituted for it. The other 282 archive blobs were
inventoried but not searched internally, and branch history was searched by
selected commit subject only. Candidate names and archived receipts alone do
not prove stage completion or replay.

Both receipt and bounded-metadata scans reproduced byte-for-byte in a second
read. Recorded Git blob IDs, compressed archive SHA-256 hashes, frozen head
refs, scripts and results enable follow-up. Tree and archive scans used
GIT_NO_LAZY_FETCH=1. A preliminary commit-availability probe omitted this
setting and may have fetched missing commits implicitly; its exact transfer is
unknown. No archive was extracted, and no checkpoint or source file was
modified by the scans. The 119-head snapshot predates subsequent evidence
commits. The search remains open; next inspect the relevant older commit trees
and generic archives using stage and provenance criteria before any restore.
