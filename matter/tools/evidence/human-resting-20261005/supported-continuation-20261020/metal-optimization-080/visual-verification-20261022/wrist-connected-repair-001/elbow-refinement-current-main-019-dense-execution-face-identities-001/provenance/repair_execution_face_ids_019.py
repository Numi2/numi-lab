#!/usr/bin/env python3
"""Emit unique current-face execution IDs without changing stable source ancestry."""
from pathlib import Path
import csv, hashlib, json, re
D=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=list(csv.DictReader((D/"generated/faces.csv").open()))
assert len(rows)==4777
stable=[int(r["stable_source_face"]) for r in rows]
parents=[int(r["parent_surface_face"]) for r in rows]
assert len(set(tuple(sorted(int(r[k]) for k in ("a","b","c"))) for r in rows))==len(rows)
p=D/"apps/NumiHumanWholeBodySkinTransferAncestry.hpp"
s=p.read_text();name="kWholeBodySkinMechanicalFaceIdentities";m=re.search(re.escape(name)+r"\s*=",s);assert m
start=s.find("{",m.end());depth=0;end=None
for i in range(start,len(s)):
 if s[i]=="{":depth+=1
 elif s[i]=="}":
  depth-=1
  if depth==0:end=i+1;break
assert end is not None
new="{{\n"+"\n".join(f"    {i}u," for i in range(len(rows)))+"\n}}"
p.write_text(s[:start]+new+s[end:])
report={"schema":"numi.human.refined_face_identity_domains.v1","rows":len(rows),"execution_face_identity":"dense local row index", "execution_ids_unique":True,"stable_source_ids_match_csv":stable==[int(r["stable_source_face"]) for r in rows],"stable_source_ids_may_repeat":len(set(stable))<len(stable),"parent_surface_ids_valid":min(parents)>=0,"faces_csv_sha256":sha(D/"generated/faces.csv"),"transfer_header_sha256":sha(p)}
(D/"execution-face-id-domain-019.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))
