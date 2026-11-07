import json
from pathlib import Path
import tempfile
from gatecheck import compare, CONTRACT

with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    record=dict(CONTRACT, failed_environment_steps=0, first_termination_step_by_environment=[3]*8,
                first_termination_tilt_rad_by_environment=[1.2]*8)
    rows=['header']+[str(i) for i in range(1,1001)]
    def put(name, result, trace):
        (root/(name+'.stdout')).write_text(json.dumps(result))
        (root/(name+'.trace.tsv')).write_text('\n'.join(trace)+'\n')
    put('one',record,rows); put('two',record,rows)
    assert compare(root,'one','two',True)['trace_mismatch_count']==0
    changed=rows.copy(); changed[4]='different'; put('two',record,changed)
    assert compare(root,'one','two',False)['trace_mismatch_count']==0
    assert compare(root,'one','two',True)['first_mismatch_line']==4
    changed[3]='different'; put('two',record,changed)
    assert compare(root,'one','two',False)['first_mismatch_line']==3
    altered=dict(record,first_termination_tilt_rad_by_environment=[2.0]*8)
    put('two',altered,rows)
    assert 'first_termination_tilt_rad_by_environment' in compare(root,'one','two',True)['differences']
    for altered in (dict(record,failed_environment_steps=1),dict(record,action_fingerprint='0')):
        put('two',altered,rows)
        try: compare(root,'one','two',True)
        except ValueError: pass
        else: raise AssertionError('must reject')
print('parity gates: equality, boundaries, mismatch, contract and physics failure passed')
