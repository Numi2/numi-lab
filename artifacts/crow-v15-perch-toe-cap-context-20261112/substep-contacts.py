from pathlib import Path
import gzip,json
p=Path("artifacts/crow-v15-rear-toe-damping-perch-screen-20261112");caps=json.loads((p/"cap-audit.json").read_text())["caps"];out={}
for arm in ("baseline","double","triple"):
 wanted={(x["interval"],x["substep"],x["lane"]) for x in caps[arm]};found={}
 with gzip.open(p/arm/"substeps.log.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_TWO_TOE_SUBSTEP "):continue
   z=line.split();key=tuple(map(int,z[1:4]))
   if key in wanted:
    assert key not in found and int(z[4])<=int(z[5]);v=list(map(float,z[6:]));assert len(v)==37;found[key]={"right_rear_proximal_normal_impulse":v[19],"right_rear_distal_normal_impulse":v[20],"right_foot_total_normal_impulse":v[3],"right_foot_vertical_impulse":v[1]}
 assert set(found)==wanted
 out[arm]=[{"interval":t,"substep":k,"lane":e,**found[t,k,e]} for t,k,e in sorted(wanted)]
result={"arms":out,"scope":"Accepted contact and velocity sampled on the same 0.000625 s physics substep of the constructed-perch bench. Normal impulses are not isolated joint torque and do not establish causation or a full landing."};Path("artifacts/crow-v15-perch-toe-cap-context-20261112/substep-contacts.json").write_text(json.dumps(result,indent=2)+"\n")
print({a:{"caps":len(xs),"toe_contact_positive":sum(x["right_rear_proximal_normal_impulse"]+x["right_rear_distal_normal_impulse"]>0 for x in xs)} for a,xs in out.items()})
