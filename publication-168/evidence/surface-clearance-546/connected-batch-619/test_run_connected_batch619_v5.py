from __future__ import annotations
import copy
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SRC=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run_connected_batch619_v5.py')
spec=importlib.util.spec_from_file_location('batch619v5',SRC)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
ACTUAL_FIT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run-002/round-001/cut-merge-001/fit')
ACTUAL_BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/connected-batch-619/run-001/round-001/retention-582/candidate.npz')

def test_complete_path_is_unchanged_and_partial_is_opt_in(tmp_path):
    base=tmp_path/'base.npz';candidate=tmp_path/'complete.npz'
    np.savez(base,points_world_m=np.zeros((3,3),np.float32),x=np.arange(3,dtype=np.int64))
    np.savez(candidate,points_world_m=np.ones((3,3),np.float32),x=np.arange(3,dtype=np.int64))
    fit_report={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED',
                'inputs':{'base_candidate_sha256':m.sha(base)},'candidate_path':str(candidate)}
    got,meta=m.select_fit_candidate(tmp_path,base,fit_report)
    assert got==candidate and meta=={'candidate_kind':'complete','fit_status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED'}
    partial={'status':'LP_NO_PROPOSAL','inputs':{'base_candidate_sha256':m.sha(base)}}
    with pytest.raises(ValueError,match='--allow-certified-partial'):
        m.select_fit_candidate(tmp_path,base,partial)

def test_actual_saved_partial_is_read_only_receipt_validated():
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    candidate,meta=m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,fit_report,allow_certified_partial=True)
    assert candidate==ACTUAL_FIT/'raw-joint-capture-563/raw-direction-001/candidate.npz'
    assert m.sha(candidate)=='9e118c3a4850904afbdc9dad243f362f6ee4bbe308d4435461ce516df67462d4'
    assert meta['candidate_kind']=='certified_partial'
    assert meta['partial_timeout']['completed_components']==13
    assert meta['partial_timeout']['failed_component']==7
    assert meta['partial_timeout']['zeroed_components']==[7]
    assert meta['partial_timeout']['global_lp_converged'] is False
    assert meta['partial_timeout']['global_l1_optimality_claim'] is False
    assert meta['partial_timeout']['retained_gain_budget_claimed'] is False

def test_missing_or_forged_partial_receipt_fails_closed():
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    missing=copy.deepcopy(fit_report);missing.pop('full_coordinate_partial_timeout564')
    with pytest.raises(ValueError,match='missing 564 partial-timeout receipt'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,missing,allow_certified_partial=True)
    forged=copy.deepcopy(fit_report)
    forged['full_coordinate_partial_timeout564']['partial_timeout_zero_fallback_opt_in']=False
    with pytest.raises(ValueError,match='partial_timeout_zero_fallback_opt_in'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,forged,allow_certified_partial=True)

def test_forged_hard_row_certificate_fails_closed(monkeypatch):
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    direction_path=ACTUAL_FIT/'raw-joint-capture-563/raw-direction-001/report.json'
    original=m.readj
    def altered_readj(path):
        value=original(path)
        if Path(path).resolve()==direction_path.resolve():
            value=copy.deepcopy(value)
            value['solver_receipt']['partial_fallback_receipt']['global_hard_rows_pass']=False
        return value
    monkeypatch.setattr(m,'readj',altered_readj)
    with pytest.raises(ValueError,match='global_hard_rows_pass'):
        m.select_fit_candidate(ACTUAL_FIT,ACTUAL_BASE,fit_report,allow_certified_partial=True)

def test_stale_or_wrong_base_is_rejected(tmp_path):
    fit_report=m.readj(ACTUAL_FIT/'report.json')
    with np.load(ACTUAL_BASE,allow_pickle=False) as z:
        arrays={k:np.array(z[k],copy=True) for k in z.files}
    arrays['points_world_m'][0,0]+=np.float32(1e-4)
    wrong=tmp_path/'wrong-base.npz';np.savez_compressed(wrong,**arrays)
    with pytest.raises(ValueError,match='base hash mismatch'):
        m.select_fit_candidate(ACTUAL_FIT,wrong,fit_report,allow_certified_partial=True)

def test_v4_source_is_unchanged_and_explicitly_pinned():
    assert m.sha(m.FROZEN_V4)==m.FROZEN_V4_SHA
