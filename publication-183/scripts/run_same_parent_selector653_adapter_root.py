#!/usr/bin/env python3
"""Adapt 370 to the newer singleton immediate-parent CSR without discarding it."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
w=Path(__file__).resolve().parent;c=w/"combined-43-57-65smooth-109-124-001";candidate=c/"candidate.npz";report=c/"audit-159/report.json";out=w/"same-parent-bone-selector653-adapt-001";path=w.parent.parent/"topology/round-037-v3-from2102/sources/select_bone_reducing_flips_370.py"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(data):
 f=data["source_O_faces"];ptr=data["refinement_face_parent_ancestry_indptr"];rows=data["refinement_face_parent_ancestry_face_rows"];parent=data["refinement_face_parent_face_row"]
 if not (np.array_equal(ptr,np.arange(len(f)+1,dtype=ptr.dtype)) and np.array_equal(rows,parent)):raise ValueError("immediate parent CSR is not a singleton equal to each face parent")
assert sha(candidate)=="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af" and not out.exists()
with np.load(candidate,allow_pickle=False) as z:validate(z)
spec=importlib.util.spec_from_file_location("selector370_adapt653",path);owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner);oldload=owner.load_pinned_runtime
def load_with_singleton_policy():
 flip,surface,bank,exact,hashes=oldload();flip.CSR_VALUE_KEYS.add("refinement_face_parent_ancestry_face_rows");old=flip.propose_flip_batch
 def checked(data,*args,**kwargs):
  validate(data);result,receipt=old(data,*args,**kwargs);validate(result)
  if not np.array_equal(result["refinement_face_parent_ancestry_indptr"],data["refinement_face_parent_ancestry_indptr"]) or not np.array_equal(result["refinement_face_parent_ancestry_face_rows"],data["refinement_face_parent_ancestry_face_rows"]):raise ValueError("immediate-parent CSR changed unexpectedly")
  return result,receipt
 flip.propose_flip_batch=checked;return flip,surface,bank,exact,hashes
owner.load_pinned_runtime=load_with_singleton_policy
sys.argv=[str(path),"--candidate",str(candidate),"--baseline-159-report",str(report),"--expected-candidate-sha256",sha(candidate),"--expected-baseline-report-sha256",sha(report),"--output-dir",str(out),"--max-flips","136"]
owner.main()
with np.load(out/"candidate.npz",allow_pickle=False) as z:validate(z)
(out/"adapter-policy.json").write_text(json.dumps({"adapter_sha256":sha(__file__),"owner_sha256":sha(path),"baseline_sha256":sha(candidate),"output_sha256":sha(out/"candidate.npz"),"policy":"face parent ancestry CSR is singleton identical to immediate face-parent row before and after; only same-parent flips are allowed; CSR remains bitwise unchanged", "physical_admission":False},indent=2)+"\n")
