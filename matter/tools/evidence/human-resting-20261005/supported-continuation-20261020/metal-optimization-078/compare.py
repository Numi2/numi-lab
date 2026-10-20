import json,pathlib,sys,math
p=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078")
a=json.load(open(p/sys.argv[1]/"summary.json"));b=json.load(open(p/sys.argv[2]/"summary.json"))
fail=[];rows=[]
for k,x in a["terminal_metrics"].items():
 y=b["terminal_metrics"].get(k)
 if y is None:fail.append(k+":missing");continue
 x=x if isinstance(x,list) else [x];y=y if isinstance(y,list) else [y]
 ok=len(x)==len(y) and all(math.isfinite(u) and math.isfinite(v) and abs(u-v)<=max(1e-7,.01*max(abs(u),abs(v))) for u,v in zip(x,y))
 rows.append(dict(metric=k,baseline=x,candidate=y,passed=ok))
 if not ok:fail.append(k)
for name,s in [("baseline",a),("candidate",b)]:
 if s["exit_code"]!=0 or not s["fem_terminal_passed"] or not s["fascia_terminal_passed"]:fail.append(name+":execution")
out=dict(baseline=sys.argv[1],candidate=sys.argv[2],passed=not fail,failures=fail,metrics=rows,speedup=float(a["body"]["wall_s"])/float(b["body"]["wall_s"]))
print(json.dumps(out,indent=2))
