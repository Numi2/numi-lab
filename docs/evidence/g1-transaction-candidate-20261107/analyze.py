import collections,csv,json
groups=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
 for r in csv.DictReader(f,delimiter="\t"):
  groups[(r["arm"],int(r["phase"]),r["array"])][int(r["index"])]=int(r["bits"])
out={}
for kind,stride in (("track",48),("frameDepth",144),("frameIdentity",576),("frameValidity",144),("camera",16)):
 previous=groups[("failure",2,kind)];failed=groups[("failure",3,kind)];control=groups[("control",3,kind)]
 for env in (0,1):
  indices=range(env*stride,(env+1)*stride)
  out[f"{kind}_env_{env}"]={"changed_from_previous":sum(previous[i]!=failed[i] for i in indices),"different_from_control":sum(failed[i]!=control[i] for i in indices)}
hist=collections.defaultdict(dict)
with open("instrumented.stderr") as f:
 for line in f:
  fields=line.split()
  if fields and fields[0]=="actorbits":
   hist[(fields[1],int(fields[2]),int(fields[3]))][int(fields[4])]=int(fields[5])
for env in (0,1):
 a=hist[("failure",2,env)];b=hist[("failure",3,env)];c=hist[("control",3,env)]
 out[f"actor_history_env_{env}"]={"changed_from_previous":sum(a[i]!=b[i] for i in range(140)),"different_from_control":sum(b[i]!=c[i] for i in range(140))}
rows=collections.defaultdict(dict)
with open("instrumented.stdout",newline="") as f:
 for r in csv.DictReader(f,delimiter="\t"):
  rows[(r["arm"],int(r["step"]),int(r["env"]))][int(r["actor_component"])]=r
for env in (0,1):
 a=rows[("failure",2,env)];b=rows[("failure",3,env)];c=rows[("control",3,env)]
 out[f"actor_output_env_{env}"]={"changed_from_previous_text":sum(a[i]["value"]!=b[i]["value"] for i in range(140)),"different_from_control_text":sum(b[i]["value"]!=c[i]["value"] for i in range(140))}
out["statuses"]={f"{a}_{s}_{e}":{k:v for k,v in next(iter(values.values())).items() if k in ("status","physics_error","done")} for (a,s,e),values in rows.items()}
out["instrumented_equals_no_frame_snapshot_text"]=open("instrumented.stdout","rb").read()==open("unobserved.stdout","rb").read()
print(json.dumps(out,indent=2))
