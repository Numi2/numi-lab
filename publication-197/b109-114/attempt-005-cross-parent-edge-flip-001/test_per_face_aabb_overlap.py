#!/usr/bin/env python3
"""Small deterministic synthetic regression for the per-face AABB predicate."""
from pathlib import Path
import hashlib,json
import numpy as np
from per_face_aabb_overlap import overlapping_rows
D=Path(__file__).resolve().parent
REPORT=D/"aabb-predicate-test-report.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lo=np.asarray([[.1,.1,.1],[1.0,.2,.2],[1.1,.2,.2],[.2,1.1,.2],[-1.,-1.,-1.],[1.,1.,1.]],np.float64)
    hi=np.asarray([[.2,.2,.2],[1.1,.3,.3],[1.2,.3,.3],[.3,1.2,.3],[2.,2.,2.],[2.,2.,2.]],np.float64)
    qlo=np.asarray([0.,0.,0.]);qhi=np.asarray([1.,1.,1.])
    actual=overlapping_rows(lo,hi,qlo,qhi).tolist();expected=[0,1,4,5]
    # The old scalar reduction loses all rows when any face is disjoint.
    old_scalar=bool(np.all(lo<=qhi))
    passed=actual==expected and old_scalar is False
    report={"schema":"numi.human.per_face_aabb_overlap.synthetic_test.v1","status":"PASS" if passed else "FAIL","query_lower":qlo.tolist(),"query_upper":qhi.tolist(),"face_lower_bounds":lo.tolist(),"face_upper_bounds":hi.tolist(),"expected_rows":expected,"actual_rows":actual,"inclusive_boundary_contacts_retained":1 in actual and 5 in actual,"disjoint_rows_excluded":2 not in actual and 3 not in actual,"old_unreduced_scalar_is_false":old_scalar,"old_scalar_mask_would_drop_all_rows":old_scalar is False,"predicate":"all(face_upper >= query_lower, axis=1) & all(face_lower <= query_upper, axis=1)","helper_path":str(D/"per_face_aabb_overlap.py"),"helper_sha256":sha(D/"per_face_aabb_overlap.py"),"test_path":str(Path(__file__).resolve()),"test_sha256":sha(__file__),"geometry_audit_run":False}
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=="__main__":main()
