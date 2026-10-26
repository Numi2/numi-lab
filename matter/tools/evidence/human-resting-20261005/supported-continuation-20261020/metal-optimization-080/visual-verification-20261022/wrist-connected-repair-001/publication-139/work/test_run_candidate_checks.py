from pathlib import Path
import copy,importlib.util,json
import numpy as np
import pytest
P=Path(__file__).with_name("run_candidate_checks.py")
s=importlib.util.spec_from_file_location("focused_checks_test",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def actual():
 return json.loads(m.BASE159.read_text())

def test_actual_538_report_and_complete_typed_domain():
 n,counts=m.validate_banks(actual(),m.BASE_SHA)
 assert n==2981 and len(counts)==185

def test_stale_candidate_and_duplicate_typed_bank_fail():
 j=actual()
 with pytest.raises(ValueError,match="stale"):m.validate_banks(j,"wrong")
 j["banks"][1]=copy.deepcopy(j["banks"][0])
 with pytest.raises(ValueError,match="duplicate"):m.validate_banks(j,m.BASE_SHA)

def test_missing_active_bank_and_wrong_pair_count_fail():
 j=actual();i=next(i for i,b in enumerate(j["banks"]) if b.get("result") and b["result"]["count"])
 bad=copy.deepcopy(j);bad["banks"][i]["result"]=None
 with pytest.raises(ValueError,match="active bank"):m.validate_banks(bad,m.BASE_SHA)
 bad=copy.deepcopy(j);bad["banks"][i]["result"]["count"]+=1
 with pytest.raises(ValueError,match="count mismatch"):m.validate_banks(bad,m.BASE_SHA)

def test_bitwise_nan_lineage_is_not_numerical_equality():
 a=np.array([np.nan,1.],dtype=np.float64)
 assert m.same(a,a.copy())
 assert not m.same(a,a.astype(np.float32))

def test_boundary_and_nonmanifold_gate():
 f=np.array([[0,1,2],[1,0,3]],dtype=np.uint32)
 assert m.boundary(f)=={(0,2),(1,2),(0,3),(1,3)}
 with pytest.raises(ValueError,match="nonmanifold"):m.boundary(np.vstack([f,[[0,1,4]]]))
