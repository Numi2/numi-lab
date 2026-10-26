import ast
from pathlib import Path
import pytest

SOURCE=Path(__file__).with_name('refresh_self_witnesses_409.py')
tree=ast.parse(SOURCE.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='count_surface_pairs')
ns={}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
count_surface_pairs=ns['count_surface_pairs']

def report_with_null_bank(active=(0,0)):
    banks=[]
    for i in range(185):
        banks.append({'body':i,'stable_bone_id':i+1,'skin_aabb_active':0,'bank_aabb_active':0,'result':None})
    banks[0]['skin_aabb_active'],banks[0]['bank_aabb_active']=active
    return {'complete':True,'bank_count':185,'expected_bank_count':185,'banks':banks}

def test_counts_null_result_only_for_inactive_bank():
    assert count_surface_pairs(report_with_null_bank())==0

def test_counts_explicit_pair_rows():
    r=report_with_null_bank()
    r['banks'][1]['result']={'count':2,'triangle_pairs':[[4,8],[7,9]]}
    assert count_surface_pairs(r)==2

def test_rejects_incomplete_audit():
    r=report_with_null_bank(); r['complete']=False
    with pytest.raises(ValueError,match='incomplete'):
        count_surface_pairs(r)

def test_rejects_null_result_for_active_bank():
    with pytest.raises(ValueError,match='active AABBs'):
        count_surface_pairs(report_with_null_bank((1,0)))

def test_rejects_mismatched_explicit_pair_count():
    r=report_with_null_bank(); r['banks'][3]['result']={'count':1,'triangle_pairs':[]}
    with pytest.raises(ValueError,match='does not match'):
        count_surface_pairs(r)

def test_rejects_duplicate_typed_bank():
    r=report_with_null_bank(); r['banks'][1]['body']=0; r['banks'][1]['stable_bone_id']=1
    with pytest.raises(ValueError,match='duplicate typed'):
        count_surface_pairs(r)
