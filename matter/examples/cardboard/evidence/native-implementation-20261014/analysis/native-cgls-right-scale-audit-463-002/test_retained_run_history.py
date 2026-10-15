#!/usr/bin/env python3
"""Integration checks for the already-retained 463/459/453/455 failed prefixes."""
import importlib.util
from pathlib import Path
script=Path(__file__).with_name('audit_chain.py')
spec=importlib.util.spec_from_file_location('audit_history_tested',script)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
CASE=Path('/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-primary48-463')

def test_full_attempted_history_and_physical_prefix():
    report=mod.run_history_comparison(CASE)
    assert report['attempted_rows']==4
    assert report['accepted_steps']==1
    assert report['attempted_step1_bent_status']=='rejected'
    assert set(report['baselines'])=={'459','453','455'}
    for baseline in report['baselines'].values():
        assert baseline['exact_rows_vs_463']==[0,1,3]
        assert baseline['physical_output_files_byte_exact']
        assert baseline['step1_bent_attempt']['native463']['step_accepted']=='0'
        assert baseline['step1_bent_attempt']['baseline']['step_accepted']=='0'
        assert baseline['step1_bent_attempt']['native463']['fgmres_iterations']=='48'
        assert baseline['step1_bent_attempt']['baseline']['fgmres_iterations']=='44'

if __name__=='__main__':
    test_full_attempted_history_and_physical_prefix()
    print('PASS 1 retained failed-prefix integration test')
