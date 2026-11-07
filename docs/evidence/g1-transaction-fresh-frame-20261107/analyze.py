import collections,csv,json,struct
def load(path):
 g=collections.defaultdict(dict)
 with open(path,newline="") as f:
  for r in csv.DictReader(f,delimiter="\t"):g[(r["arm"],int(r["phase"]),r["array"])][int(r["index"])]=int(r["bits"])
 return g
baseline=load("baseline-snapshot.tsv");candidate=load("candidate-snapshot.tsv")
out={}
for name,stride in (("track",48),("frameDepth",144),("frameIdentity",576),("frameValidity",144),("camera",16)):
 for env in (0,1):
  a=baseline[("failure",2,name)];b=baseline[("failure",3,name)];c=candidate[("failure",2,name)];d=candidate[("failure",3,name)]
  ix=range(env*stride,(env+1)*stride)
  out[f"{name}_env_{env}"]={"baseline_step2_to_3":sum(a[i]!=b[i] for i in ix),"candidate_step2_to_3":sum(c[i]!=d[i] for i in ix),"pre_step_baseline_vs_candidate":sum(a[i]!=c[i] for i in ix),"candidate_vs_baseline_step3":sum(d[i]!=b[i] for i in ix)}
hist=collections.defaultdict(dict)
for line in open("candidate.stderr"):
 parts=line.split()
 if parts and parts[0]=="actorbits":hist[(parts[1],int(parts[2]),int(parts[3]))][int(parts[4])]=int(parts[5])
for env in (0,1):
 a=hist[("failure",2,env)];b=hist[("failure",3,env)];c=hist[("control",3,env)]
 out[f"actor_history_env_{env}"]={"changed_from_step2":sum(a[i]!=b[i] for i in range(140)),"different_from_control_step3":sum(b[i]!=c[i] for i in range(140))}
out["target_confidence"]={str(e):struct.unpack("<f",struct.pack("<I",candidate[("failure",2,"track")][e*48+43]))[0] for e in (0,1)}
out["candidate_text_equals_no_snapshot"]=open("candidate.stdout","rb").read()==open("no_snapshot.stdout","rb").read()
print(json.dumps(out,indent=2))
