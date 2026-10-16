#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
P=Path('/Users/home/cardboard-implementation-20261014/analysis/publication029-independent-byte-audit-001')
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
E=R/'matter/examples/cardboard/evidence/native-implementation-20261014'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def info(p): return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
report=json.loads((P/'report.json').read_text())
inputs={
 'publication_binding029':info(E/'publication-binding-reference029.json'),
 'summary_current':info(E/'summary.json'),
 'summary_reference028':info(E/'summary-reference028.json'),
 'publication_binding028':info(E/'publication-binding-reference028.json'),
 'pathspec_reference028':info(B/'publication-reference028-pathspec.nul'),
 'pathspec_reference029':info(B/'publication-reference029-pathspec.nul'),
 'staged_pathlist029':info(B/'publication029-staged-paths.nul'),
 'staged_source_check_report':info(B/'analysis/publication029-staged-source-check-001/report.json'),
 'source168_frozen_manifest':info(B/'snapshots/source-reference-jacobian-168/frozen-source.json'),
 'audit_report':info(P/'report.json'),
 'audit_script':info(P/'audit.py'),
 'payload_audit_jsonl':info(P/'payload-audit.jsonl'),
 'readme':info(P/'README.md'),
 'final_stdout':info(P/'audit-final.stdout.log'),
 'final_stderr':info(P/'audit-final.stderr.log'),
 'initial_attempt_stderr':info(P/'audit.stderr.log'),
 'initial_attempt_note':info(P/'initial-attempt-note.md'),
}
manifest={
 'schema':'numi.cardboard.publication029_independent_byte_audit_manifest.v1',
 'verdict':report['verdict'],
 'read_only':True,
 'repository':str(R),
 'head':report['head'],
 'staged_path_count':report['staged_path_count'],
 'staged_pathlist_sha256':report['staged_pathlist_sha256'],
 'summary_sha256':report['summary_sha256'],
 'publication_binding_sha256':report['binding_sha256'],
 'source168_frozen_sha256':report['source168']['frozen_source_sha256'],
 'new_payload_count':report['payloads']['record_count'],
 'all_staged_blobs_match_worktree':report['all_staged_blob_bytes_match_worktree'],
 'all_payload_originals_and_restores_match':report['payloads']['all_external_originals_verified'] and report['payloads']['all_staged_restore_bytes_verified'],
 'inputs_and_outputs':inputs,
 'retained_initial_failure':'first audit attempt looked for physical_validation in summary.json; corrected final run reads publication binding; initial log preserved',
}
out=P/'manifest.json'; out.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'manifest':str(out),'sha256':sha(out),'report_sha256':sha(P/'report.json')},sort_keys=True))
