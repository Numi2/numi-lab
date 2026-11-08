#!/usr/bin/env python3
"""Inject a comparison failure into a temporary trajectory and check stop semantics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
p=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as name:
    d=Path(name)
    for item in ('solve.py','tangent.py','solve-cases.json'):
        (d/item).symlink_to(p/item)
    text=(p/'trajectory.py').read_text()
    old="""                row={'case':index,'step':step,'candidate_status':a['status'],'reference_status':b['status'],'candidate_iterations':"""
    new="""                if index==1 and step==1: pass_step=False  # test-only injected gate failure
                row={'case':index,'step':step,'candidate_status':a['status'],'reference_status':b['status'],'candidate_iterations':"""
    assert text.count(old)==1
    (d/'trajectory.py').write_text(text.replace(old,new))
    run=subprocess.run([sys.executable,str(d/'trajectory.py'),sys.argv[1]],capture_output=True,text=True,timeout=60)
    result=json.loads((d/'trajectory-analysis.json').read_text())
    counts={str(i):sum(r.get('case')==i for r in result['cases']) for i in (1,2,3)}
    assert run.returncode==1 and not result['all_pass'] and counts=={'1':1,'2':10,'3':10}, (run.returncode,counts)
    assert sum(not r['passed'] for r in result['cases'])==1
    print('injected_gate_failure=detected exit=1 case_steps='+json.dumps(counts,sort_keys=True))
