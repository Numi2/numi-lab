from pathlib import Path
import sys
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import combined_bank_coverage663 as mod

def gm(face,body,bone,tri,corner,node,lower=0.,anchor=False):
    return {"face_row":face,"body":body,"bone_id":bone,"triangle_id":tri,
            "corner_index":corner,"node_row":node,"lower_mm":lower,"anchor_zero_coefficients":anchor}
def sm(face,body,bone,tri,corner,node,selected=True):
    ident=[face,body,bone,tri]
    return {"kind":"face","face_row":face,"body":body,"bone_id":bone,"triangle_id":tri,
            "corner_index":corner,"node_row":node,"selector_pair_selected":selected,"selector_identity":ident}
def fixture(include_all=True):
    req={(1,43,16,4),(2,109,114,5)}
    guards=[];meta=[];h=[];w=[]
    for face,body,bone,tri in sorted(req):
        for c in range(3):
            lo=.2 if c==0 else (-.1 if c==1 else 0.)
            guards.append(gm(face,body,bone,tri,c,face*3+c,lo))
            if lo>1e-12:meta.append(sm(face,body,bone,tri,c,face*3+c));h.append(lo);w.append(.5)
    if not include_all:
        meta.pop();h.pop();w.pop()
    return {"C":sparse.csr_matrix((len(meta),6)),"h":h,"weights":w,"metadata":meta,"guard_metadata":guards},req

def test_complete_guard_to_soft_pair_coverage_and_three_corners():
    value,req=fixture()
    r=mod.validate_rows(value,req)
    assert r["required_pair_count"]==2 and r["covered_pair_count"]==2
    assert r["guard_corner_count"]==6 and r["actual_positive_soft_corner_rows"]==2
    assert r["all_requested_pairs_have_positive_soft_rows"]

def test_missing_expected_soft_corner_and_subthreshold_positive_are_not_passed():
    value,req=fixture()
    value["metadata"].pop();value["h"].pop();value["weights"].pop()
    value["C"]=sparse.csr_matrix((len(value["metadata"]),6))
    r=mod.validate_rows(value,req)
    assert r["missing_expected_soft_corners"]
    assert not r["all_requested_pairs_have_positive_soft_rows"]
    value,req=fixture()
    value["guard_metadata"][1]["lower_mm"]=5e-13
    r=mod.validate_rows(value,req)
    assert len(r["subthreshold_positive_corners"])==1
    assert not r["all_requested_pairs_have_positive_soft_rows"]

def test_unselected_requested_pair_fails_closed():
    value,req=fixture()
    value["metadata"][0]["selector_pair_selected"]=False
    with pytest.raises(ValueError,match="selected 638"):
        mod.validate_rows(value,req)

def test_guard_domain_and_anchor_positive_reject():
    value,req=fixture()
    value["guard_metadata"].pop()
    with pytest.raises(ValueError,match="three 458 guard"):
        mod.validate_rows(value,req)
    value,req=fixture()
    value["guard_metadata"][0]["anchor_zero_coefficients"]=True
    r=mod.validate_rows(value,req)
    assert r["anchored_positive_corners"]
    assert not r["all_requested_pairs_have_positive_soft_rows"]

