import json,re,sys
from pathlib import Path
w=Path(sys.argv[1]);root=Path(sys.argv[2]);ed=root/'docs/evidence'
now=[json.loads(x) for x in (w/'archive-content.jsonl').read_text().splitlines()]
prior=[json.loads(x) for x in (ed/'suture-generic-archive-scan-20261108/results.jsonl').read_text().splitlines()]
prior+=json.loads((ed/'suture-generic-archive-scan-20261108/zstd-results.json').read_text())
prior+=json.loads((ed/'suture-branch-tip-search-20261108/surgical-archive-members.json').read_text())['archives']
ref={x['object']:x['sha256'] for x in prior if 'sha256' in x}
links=json.loads((ed/'suture-generic-archive-scan-20261108/link-analysis.json').read_text())
inventory=json.loads((ed/'suture-branch-tip-search-20261108/archive-inventory.json').read_text())['archive_blobs']
expected={x['object'] for x in inventory}-{x['object'] for x in links}
observed={x['object'] for x in now}
sha_mismatch=[x['object'] for x in now if ref.get(x['object'])!=x.get('compressed_sha256')]
assert len(now)==len(observed) and observed==expected
assert all(x['complete'] and not x['errors'] for x in now)
assert not sha_mismatch,sha_mismatch
schema=re.compile(rb'(?m)^schema\tnumi\.dual-psm-suture-handoff-state\.v3\r?$');phase=re.compile(rb'(?m)^phase\ttissue-receiver-bridge-start\r?$')
assert schema.search(b'schema\tnumi.dual-psm-suture-handoff-state.v3\n')
assert phase.search(b'phase\ttissue-receiver-bridge-start\n')
assert not phase.search(b'// phase\ttissue-receiver-bridge-start\n')
classified=json.loads((w/'classified-schema-matches.json').read_text())
summary={'archives':len(now),'links_skipped':len(links),'prior_sha_comparisons':len(ref),'sha_mismatches':sha_mismatch,'members':sum(x['members'] for x in now),'regular_prefixes':sum(x['regular_prefixes'] for x in now),'schema_matches':sum(m['schema'] for x in now for m in x['matches']),'phase_matches':sum(m['phase'] for x in now for m in x['matches']),'classification_errors':classified['errors'],'compressed_bytes':sum(x['bytes_read'] for x in now),'positive_and_negative_regex_controls':'passed'}
(w/'archive-audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
