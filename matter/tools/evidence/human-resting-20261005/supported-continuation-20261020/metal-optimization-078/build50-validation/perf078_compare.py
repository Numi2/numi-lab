import json,pathlib,sys
p=pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078")
a,b,out=sys.argv[1:4]
x=json.load(open(p/a/"summary.json"));y=json.load(open(p/b/"summary.json"));rows=[]
for k,v in x["terminal_metrics"].items():
 w=y["terminal_metrics"].get(k)
 if w is None:rows.append(dict(metric=k,passed=False,reason="missing"));continue
 av=v if isinstance(v,list) else [v];bv=w if isinstance(w,list) else [w]
 err=max(abs(i-j)/max(abs(i),abs(j),1e-5) for i,j in zip(av,bv))
 rows.append(dict(metric=k,baseline=v,candidate=w,relative_error=err,passed=err<=.01))
result=dict(baseline=a,candidate=b,passed=bool(rows) and all(v["passed"] for v in rows),speedup=float(x["body"]["wall_s"])/float(y["body"]["wall_s"]),metrics=rows)
json.dump(result,open(out,"w"),indent=2)
print(json.dumps({k:v for k,v in result.items() if k!="metrics"}))
print([r for r in rows if not r["passed"]])
