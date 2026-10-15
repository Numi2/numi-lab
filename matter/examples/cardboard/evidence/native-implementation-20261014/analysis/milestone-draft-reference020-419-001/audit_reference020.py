#!/usr/bin/env python3
"""Independently verify the archived reference020 case/file index and payloads."""
from __future__ import annotations
import gzip, hashlib, json, pathlib

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
REPO=pathlib.Path('/Users/home/numi-cardboard-20261006')
EVIDENCE=REPO/'matter/examples/cardboard/evidence/native-implementation-20261014'
BINDING=EVIDENCE/'publication-binding-reference020.json'
SUMMARY=EVIDENCE/'summary.json'
OUT=BASE/'analysis/milestone-draft-reference020-419-001'
DRAFT=BASE/'analysis/operator-fd-e2-independent-adjudication-419-001'

def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def verify_gzip(p):
    h=hashlib.sha256(); n=0
    with gzip.open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b); n+=len(b)
    return n,h.hexdigest()

def main():
    binding=json.loads(BINDING.read_text())
    summary=json.loads(SUMMARY.read_text())
    assert binding['schema']=='numi.cardboard.reference020-publication-binding.v1'
    assert sha_file(SUMMARY)==binding['summary_sha256']
    records={r['case']:r for r in summary['records']}
    cases=binding['cases']
    assert len(cases)==56 and len(set(cases))==56
    copied=0; checked_direct=0; checked_gzip=0; missing_payload=0
    direct_mismatch=[]; gzip_compressed_mismatch=[]; gzip_expanded_mismatch=[]
    source_drift=[]; source_missing=[]; duplicate_targets=[]; seen_targets=set(); no_payload=[]
    for case in cases:
        assert case in records, case
        rec=records[case]
        for row in rec['files']:
            copied+=1
            rel=row['name']
            target_kind='repository_copy' if 'repository_copy' in row else 'repository_compressed_copy' if 'repository_compressed_copy' in row else None
            if target_kind is None:
                no_payload.append({'case':case,'name':rel})
                missing_payload+=1
                continue
            target_rel=row[target_kind]
            target_path=EVIDENCE/target_rel
            target_norm=target_path.resolve()
            if target_norm in seen_targets: duplicate_targets.append(str(target_norm))
            seen_targets.add(target_norm)
            if not target_path.is_file():
                missing_payload+=1
                no_payload.append({'case':case,'name':rel,'target':target_rel})
                continue
            if target_kind=='repository_copy':
                checked_direct+=1
                size=target_path.stat().st_size
                got=sha_file(target_path)
                if size!=row['bytes'] or got!=row['sha256']:
                    direct_mismatch.append({'case':case,'name':rel,'expected_bytes':row['bytes'],'actual_bytes':size,'expected_sha256':row['sha256'],'actual_sha256':got})
            else:
                checked_gzip+=1
                compressed=sha_file(target_path)
                if target_path.stat().st_size!=row.get('compressed_bytes') or compressed!=row.get('compressed_sha256'):
                    gzip_compressed_mismatch.append({'case':case,'name':rel,'target':target_rel,'expected_compressed_bytes':row.get('compressed_bytes'),'actual_compressed_bytes':target_path.stat().st_size,'expected_compressed_sha256':row.get('compressed_sha256'),'actual_compressed_sha256':compressed})
                actual_size,actual_sha=verify_gzip(target_path)
                if actual_size!=row['bytes'] or actual_sha!=row['sha256']:
                    gzip_expanded_mismatch.append({'case':case,'name':rel,'expected_bytes':row['bytes'],'actual_bytes':actual_size,'expected_sha256':row['sha256'],'actual_sha256':actual_sha})
            external=pathlib.Path(row['external_path'])
            if external.is_file():
                src_hash=sha_file(external)
                if src_hash!=row['sha256']:
                    source_drift.append({'case':case,'name':rel,'external_path':str(external),'indexed_sha256':row['sha256'],'current_external_sha256':src_hash})
            else:
                source_missing.append({'case':case,'name':rel,'external_path':str(external)})
    indexed_sum=sum(len(records[c]['files']) for c in cases)
    assert indexed_sum==850 and copied==850 and copied==binding['copied_files']
    report={
      'schema':'numi.cardboard.reference020-independent-integrity-audit.v1',
      'status':'complete_diagnostic',
      'case_count':len(cases), 'indexed_file_count':indexed_sum,
      'binding_declared_file_count':binding['copied_files'],
      'direct_payloads_verified':checked_direct,
      'gzip_payloads_verified':checked_gzip,
      'missing_payloads':missing_payload,
      'direct_mismatches':direct_mismatch,
      'gzip_compressed_mismatches':gzip_compressed_mismatch,
      'gzip_expanded_mismatches':gzip_expanded_mismatch,
      'duplicate_archive_targets':duplicate_targets,
      'current_external_source_drift_count':len(source_drift),
      'current_external_source_missing_count':len(source_missing),
      'current_external_source_drift_examples':source_drift[:20],
      'current_external_source_missing_examples':source_missing[:20],
      'all_archived_payloads_verified': not (missing_payload or direct_mismatch or gzip_compressed_mismatch or gzip_expanded_mismatch or duplicate_targets),
      'inputs':{
        'publication_binding_sha256':sha_file(BINDING),
        'summary_sha256':sha_file(SUMMARY),
        'summary_declared_sha256':binding['summary_sha256'],
        'source020_milestone':summary['milestone'],
        'new419_adjudication_manifest_sha256':sha_file(DRAFT/'manifest.json'),
        'new419_adjudication_report_sha256':sha_file(DRAFT/'report.json')
      },
      'limitations':['This verifies archive index and copied bytes, not scientific correctness or source conclusions. Current external source drift/missing files are reported separately from archived-byte integrity.']
    }
    assert report['all_archived_payloads_verified']
    (OUT/'index-audit.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report
if __name__=='__main__': main()
