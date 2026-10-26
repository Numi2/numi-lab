#!/usr/bin/env python3
"""Check post-build 130/159 evidence for a 548 candidate without promotion."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality548_verify_helper',HERE/'condition_area_quality_548.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True);ap.add_argument('--baseline-report',default=str(h.DEFAULT_SURFACE))
    ap.add_argument('--after-report',required=True);ap.add_argument('--self-summary',required=True);ap.add_argument('--self-pairs',required=True)
    a=ap.parse_args(); candidate=Path(a.candidate); after=json.loads(Path(a.after_report).read_text())
    baseline=json.loads(Path(a.baseline_report).read_text()); selfdoc=json.loads(Path(a.self_summary).read_text())
    if baseline.get('candidate_sha256')!=h.BASE_SHA: raise h.ProposalError('baseline exact report is not pinned to selected 2409 candidate')
    ch=h.sha(candidate)
    if ch!=after.get('candidate_sha256') or ch!=selfdoc.get('candidate_sha256'):
        raise h.ProposalError('candidate hash does not bind both post-build exact audits')
    if not selfdoc.get('complete_surface_audit'): raise h.ProposalError('130 self audit is incomplete')
    selfpairs=np.asarray(np.load(a.self_pairs,allow_pickle=False),dtype=np.int64).reshape((-1,2))
    if len(selfpairs): raise h.ProposalError(f'exact self audit reports {len(selfpairs)} pairs')
    delta=h.check_exact_nonincrease(baseline,after)
    result={'schema':'numi.human.area_quality_conditioner_548.audit-verification.v1',
            'status':'PASS_EXACT_ZERO_SELF_AND_185_BANK_NONINCREASE' if delta['passed'] else 'FAIL_185_BANK_INCREASE',
            'candidate_path':str(candidate),'candidate_sha256':ch,
            'baseline_report_sha256':h.sha(a.baseline_report),'after_report_sha256':h.sha(a.after_report),
            'self_summary_sha256':h.sha(a.self_summary),'self_pairs_sha256':h.sha(a.self_pairs),
            'zero_self_pairs':True,'all185_nonincrease':delta}
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if delta['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
