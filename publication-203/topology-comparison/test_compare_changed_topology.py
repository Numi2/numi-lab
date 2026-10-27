#!/usr/bin/env python3
"""Fail-closed domain/ancestry regression checks for topology comparison."""
from pathlib import Path
import importlib.util, json, tempfile, numpy as np
p=Path(__file__).parent;spec=importlib.util.spec_from_file_location("cmp",p/"compare_changed_topology.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def rejects(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError("invalid map was accepted")
m.verify_parent_pin({"pin":np.array(["abcd"])},"pin","abcd")
rejects(lambda:m.verify_parent_pin({},"pin","abcd"))
rejects(lambda:m.verify_parent_pin({"pin":np.array(["wrong"])},"pin","abcd"))
oldf=np.array([[0,1,2],[0,2,3]],dtype=np.int64);newf=np.array([[0,1,3]],dtype=np.int64)
z={"ptr":np.array([0,2],dtype=np.int64),"rows":np.array([0,1],dtype=np.int64),"node_map":np.array([0,1,-1,2],dtype=np.int64)}
assert m.make_face_parents(oldf,newf,z,False,"ptr","rows")[0].tolist()==[0,1]
assert m.make_node_map(np.zeros((4,3)),np.zeros((3,3)),z,False,"node_map").tolist()==[0,1,-1,2]
rejects(lambda:m.make_face_parents(oldf,newf,{"ptr":np.array([0,0]),"rows":np.array([],dtype=np.int64)},False,"ptr","rows"))
rejects(lambda:m.make_face_parents(oldf,newf,{"ptr":np.array([0,2]),"rows":np.array([0,0])},False,"ptr","rows"))
rejects(lambda:m.make_face_parents(oldf,newf,{"ptr":np.array([0,1]),"rows":np.array([2])},False,"ptr","rows"))
m.verify_reverse_node_map(np.array([0,1,-1,2]),{"reverse":np.array([0,1,3])},"reverse",3)
rejects(lambda:m.verify_reverse_node_map(np.array([0,1,-1,2]),{"reverse":np.array([0,3,1])},"reverse",3))
rejects(lambda:m.make_node_map(np.zeros((4,3)),np.zeros((3,3)),{"node_map":np.array([0,1,1,2])},False,"node_map"))
rejects(lambda:m.make_node_map(np.zeros((4,3)),np.zeros((3,3)),{"node_map":np.array([0,1,-1,3])},False,"node_map"))
rejects(lambda:m.check_scene({"points_world_m":np.zeros((4,3)),"source_O_faces":np.array([[0,1,2]]),"refinement_face_parent_face_row":np.array([0])}))
rejects(lambda:m.check_scene({"points_world_m":np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),"source_O_faces":np.array([[0,1,1],[0,1,2]]),"refinement_face_parent_face_row":np.array([0,1])}))
rejects(lambda:m.check_scene({"points_world_m":np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),"source_O_faces":np.array([[0,1,2]]),"refinement_face_parent_face_row":np.array([0])}))
retained,added,cleared,deleted=m.compare_inside({(2,65,99),(3,61,97)},{(2,61,97)},np.array([0,1,-1,2],dtype=np.int64))
assert retained=={(2,61,97)} and not added and not cleared and deleted==[(2,65,99)]
r=json.loads((p/"identity-parity-v7.json").read_text())
assert r["banks"]==185 and r["raw_crossings"]["before"]==r["raw_crossings"]["after"]==977
assert r["mapped_parent_pairs"]["before"]==r["mapped_parent_pairs"]["after"]==651
assert r["mapped_parent_pairs"]["added_count"]==r["mapped_parent_pairs"]["removed_count"]==0
assert r["inside"]["deleted_or_merged_old_inside_count"]==0 and not r["inside"]["retained_old_identity_added"] and not r["inside"]["retained_old_identity_removed"]
bank_report=p.parent/"combined-d071-b109-b61-001/audit-159/report.json"
j=json.loads(bank_report.read_text())
bank=next(b for b in j["banks"] if b.get("result") and b["result"]["triangle_pairs"])
bank["result"]["triangle_pairs"][0][1]=int(bank["bank_faces"])
with tempfile.TemporaryDirectory() as t:
 q=Path(t)/"bad-bank.json";q.write_text(json.dumps(j))
 rejects(lambda:m.read159(q,j["candidate_sha256"],42379))
print("PASS: real 185-bank parity; changed-face CSR, deleted-node, duplicate-node, out-of-domain, and orphan-node, degenerate-face and bone-triangle-domain gates")
