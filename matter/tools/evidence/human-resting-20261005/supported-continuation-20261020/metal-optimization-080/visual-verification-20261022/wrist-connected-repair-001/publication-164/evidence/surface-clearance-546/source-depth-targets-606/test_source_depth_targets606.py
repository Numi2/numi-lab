import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).with_name('build_source_depth_targets606.py')
spec=importlib.util.spec_from_file_location('source_depth_targets606',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def audit(statuses,candidate='abc'):
 rows=[{'query_row':i,'status':s,'current_source_outward_dot':(-.4 if s!='PASS' else .5)} for i,s in enumerate(statuses)]
 bad=sum(s!='PASS' for s in statuses)
 return {'candidate_sha256':candidate,'target_row_count':len(rows),'agreement_count':len(rows)-bad,'disagreement_count':bad,'status':'CURRENT_SOURCE_OUTWARD_DISAGREEMENT' if bad else 'PASS','rows':rows}
def test_current_disagreement_is_retained_but_not_a_gate():
 r=m.validate_current_normal_diagnostic(audit(['PASS','CURRENT_SOURCE_OUTWARD_DISAGREEMENT']),'abc',2)
 assert r['disagreement_count']==1 and r['rows']==2
def test_current_diagnostic_candidate_mismatch_rejects():
 with pytest.raises(ValueError,match='another candidate'): m.validate_current_normal_diagnostic(audit(['PASS'],'wrong'),'abc',1)
def test_current_diagnostic_duplicate_rows_rejects():
 d=audit(['PASS','PASS']);d['rows'][1]['query_row']=0
 with pytest.raises(ValueError,match='duplicate'): m.validate_current_normal_diagnostic(d,'abc',2)
def test_target_rows_must_match_all_diagnostic_rows():
 assert m.validate_row_identity([4,8],[8,4])
 with pytest.raises(ValueError,match='do not match'): m.validate_row_identity([4,8],[4,9])
def test_target_rows_must_be_unique():
 with pytest.raises(ValueError,match='unique'): m.validate_row_identity([4,4],[4,5])
