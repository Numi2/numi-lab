"""Correct provenance labels on a completed 549 projection of a transferred field.

This tool does not run or change 549's numerical code. It binds the descriptor,
raw target and 549 output, keeps an exact copy of the original report, and
replaces the 549 owner's generic raw-LP scope string with this proposal's true
source kind.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
EXPECTED_549=ROOT/'D/component-area-backtracking-549/component_area_backtracking549.py'
EXPECTED_549_SHA='523b1aef89d92eef09bf63e9de2d4f3fa6bd938ef53a45915f9ea138f5b8f249'
SCHEMA='numi.human.generic_coordinate_transfer_549_annotation565.v1'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read(path): return json.loads(Path(path).read_text())

def annotate(report_path, raw_report_path, descriptor_path):
    report_path=Path(report_path).resolve(); raw_report_path=Path(raw_report_path).resolve(); descriptor_path=Path(descriptor_path).resolve()
    if sha(EXPECTED_549)!=EXPECTED_549_SHA: raise ValueError('pinned 549 numerical owner changed')
    descriptor=_read(descriptor_path); raw=_read(raw_report_path); report=_read(report_path)
    if descriptor.get('schema')!='numi.human.generic_coordinate_transfer_for_549.v1' or descriptor.get('status')!='PREPARED_INPUT_DESCRIPTOR_ONLY_UNADMITTED':
        raise ValueError('unknown or non-prepared 565 transfer descriptor')
    if descriptor.get('not_a_raw_lp_candidate') is not True or descriptor.get('raw_lp_feasibility_claim') is not False:
        raise ValueError('descriptor must explicitly deny LP-feasibility claims')
    expected_source=descriptor.get('source_kind')
    if not expected_source or raw.get('source_kind')!=expected_source or raw.get('raw_lp_feasibility_claim') is not False:
        raise ValueError('raw input provenance does not match generic transferred-coordinate source')
    base=Path(descriptor['base_candidate']).resolve(); target=Path(descriptor['coordinate_target_candidate']).resolve()
    if sha(base)!=descriptor['base_sha256'] or sha(target)!=descriptor['coordinate_target_sha256']:
        raise ValueError('565 base or coordinate target changed since descriptor creation')
    if descriptor.get('549_compatibility',{}).get('backtracker_sha256')!=EXPECTED_549_SHA:
        raise ValueError('descriptor does not pin the expected 549 owner')
    if raw.get('base_candidate')!=str(base) or raw.get('base_sha256')!=sha(base):
        raise ValueError('generic raw input does not bind the 6ce base')
    if Path(raw.get('candidate_path','')).resolve()!=target or raw.get('candidate_sha256')!=sha(target):
        raise ValueError('generic raw input does not bind the transferred coordinate target')
    staged_candidate=raw_report_path.parent/'candidate.npz'
    if not staged_candidate.is_file() or sha(staged_candidate)!=sha(target):
        raise ValueError('549 raw-dir candidate.npz does not match the transferred coordinate target')
    if raw.get('free_nodes')!=descriptor.get('free_nodes'):
        raise ValueError('generic raw free-node rows differ from bytewise changed rows')
    if report.get('status')!='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED':
        raise ValueError('549 output is not a construction-only proposal')
    if not isinstance(report.get('full_final_gate'),dict) or not report['full_final_gate']:
        raise ValueError('549 output is missing its gate result')
    if Path(report.get('base_candidate','')).resolve()!=base or report.get('base_sha256')!=sha(base):
        raise ValueError('549 projected output base does not bind descriptor base')
    output_candidate=Path(report.get('candidate_path','')).resolve()
    if not output_candidate.is_file() or sha(output_candidate)!=report.get('candidate_sha256'):
        raise ValueError('549 projected candidate hash mismatch')
    if report.get('native_admission') is not False or report.get('accepted_physical_steps')!=0:
        raise ValueError('549 report has unexpected admission fields')
    original_bytes=report_path.read_bytes()
    original_hash=hashlib.sha256(original_bytes).hexdigest()
    backup=report_path.with_name(report_path.stem+'.pre-565-annotation'+report_path.suffix)
    receipt_path=report_path.with_name(report_path.stem+'.source-annotation-receipt.json')
    if backup.exists() or receipt_path.exists(): raise FileExistsError('refusing to overwrite prior 565 annotation evidence')
    original_scope=report.get('scope')
    report['original_549_scope_label']={
        'value':original_scope,
        'meaning':'generic owner default label retained for audit; it is not accurate source provenance for this transferred-coordinate proposal',
    }
    report['scope']='regional componentwise Float32 quality backtracking of a transferred coordinate correction; not an LP-feasibility claim; full 545 exact checks still required'
    report['source_method_annotation_565']={
        'schema':SCHEMA,
        'source_kind':expected_source,
        'source_descriptor_path':str(descriptor_path),
        'source_descriptor_sha256':sha(descriptor_path),
        'raw_input_report_path':str(raw_report_path),
        'raw_input_report_sha256':sha(raw_report_path),
        'raw_input_candidate_path':str(target),
        'raw_input_candidate_sha256':sha(target),
        'original_549_backtracker_path':str(EXPECTED_549),
        'original_549_backtracker_sha256':EXPECTED_549_SHA,
        'original_549_report_sha256':original_hash,
        'raw_lp_feasibility_claim':False,
        'full_545_exact_check_required':True,
        'native_admission':False,
        'accepted_physical_steps':0,
    }
    backup.write_bytes(original_bytes)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    receipt={
        'schema':SCHEMA,'status':'PROVENANCE_ANNOTATED_NUMERIC_549_OUTPUT_UNCHANGED',
        'annotated_report_path':str(report_path),'annotated_report_sha256':sha(report_path),
        'original_report_backup_path':str(backup),'original_report_sha256':original_hash,
        'candidate_path':str(output_candidate),'candidate_sha256':sha(output_candidate),
        'descriptor_path':str(descriptor_path),'descriptor_sha256':sha(descriptor_path),
        'raw_input_report_path':str(raw_report_path),'raw_input_report_sha256':sha(raw_report_path),
        'raw_lp_feasibility_claim':False,'full_545_exact_check_required':True,
        'native_admission':False,'accepted_physical_steps':0,
    }
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return receipt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--549-report',dest='proposal_report',type=Path,required=True)
    ap.add_argument('--raw-report',type=Path,required=True)
    ap.add_argument('--descriptor',type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(annotate(a.proposal_report,a.raw_report,a.descriptor),indent=2,sort_keys=True))
if __name__=='__main__': main()
