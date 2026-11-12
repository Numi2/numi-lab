from pathlib import Path
import json
p=Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112")
def read(name):
 out={}
 for l in (p/("layout-"+name+".stderr")).read_text().splitlines():
  if l.startswith("CROW_LAYOUT_RADIUS "):
   _,i,b,r=l.split();assert int(i) not in out;out[int(i)]=(int(b),float(r))
 return out
a=read("control");b=read("high");assert len(a)==len(b)==32 and a.keys()==b.keys();changes=[{"index":i,"control_body":a[i][0],"candidate_body":b[i][0],"control_radius_m":a[i][1],"candidate_radius_m":b[i][1]} for i in a if a[i]!=b[i]];assert all(x["control_body"]==x["candidate_body"] for x in changes)
out={"entries":len(a),"changes":changes,"scope":"The world-bound and collision-radius fingerprints genuinely differ. Layout audit separately suppresses those two ingredients only for diagnosis, and must not be used as a production policy contract. The active Metal task source reads contactMemberRadii in its link-clearance reward branch; altered geometry can change observations and behavior through other pathways even when layout tables match."};(p/"radii-comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
