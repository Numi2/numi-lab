import json,math
from pathlib import Path
p=Path(__file__).resolve().parent
def diffs(a,b,path=""):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in ("scratch","row_trace"):yield from diffs(a.get(k),b.get(k),path+"/"+k)
 elif isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):yield [path+"/length",len(a),len(b)]
  for i,(x,y) in enumerate(zip(a,b)):yield from diffs(x,y,path+"/"+str(i))
 elif a!=b:yield [path,a,b]
out={}
for n in (5,9):
 a=json.loads((p/"runs"/("rod%d-old.stdout"%n)).read_text());b=json.loads((p/"fallback"/("rod%d.stdout"%n)).read_text())
 ds=list(diffs(a,b));numeric=[abs(x[1]-x[2]) for x in ds if isinstance(x[1],(int,float)) and not isinstance(x[1],bool) and isinstance(x[2],(int,float)) and not isinstance(x[2],bool) and math.isfinite(x[1]) and math.isfinite(x[2])]
 out[str(n)]={"predeclared_exact_gate":len(ds)==0,"difference_count":len(ds),"first_differences":ds[:12],"max_numeric_difference_mixed_units":max(numeric,default=0.),"old_accepted":a["success"],"forced_accepted":b["success"],"old_status_codes":[s["code"] for s in a["steps"]],"forced_status_codes":[s["code"] for s in b["steps"]],"repeat":b["repeat"],"exact_trace":b["exact_trace"],"capture_noninvasive":b["capture_noninvasive"]}
(p/"fallback/analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
