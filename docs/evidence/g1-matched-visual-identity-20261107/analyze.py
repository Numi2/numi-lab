import collections,csv,json,math,struct
g=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
 for r in csv.DictReader(f,delimiter="\t"):g[(r["arm"],int(r["phase"]),r["array"])][int(r["index"])]=int(r["bits"])
def val(bits):return struct.unpack("<f",struct.pack("<I",bits))[0]
out={}
for arm in ("control","failure"):
 for phase in (1,2,3):
  track=g[(arm,phase,"track")];ids=g[(arm,phase,"frameIdentity")];valid=g[(arm,phase,"frameValidity")];depth=g[(arm,phase,"frameDepth")]
  for env in (0,1):
   pixels=[p-env*144 for p in range(env*144,(env+1)*144) if ids[4*p+1]==105 and ids[4*p+2]==36 and valid[p] and val(depth[p])>0 and math.isfinite(val(depth[p]))]
   out[f"{arm}_{phase}_{env}"]={"target_confidence":val(track[env*48+43]),"target_pixels":pixels,"target_track":[val(track[env*48+40+i]) for i in range(8)]}
for env in (0,1):
 a=g[("failure",2,"track")];b=g[("failure",3,"track")]
 out[f"failure_env_{env}_changes"]=[{"index_within_env":i-env*48,"before_bits":a[i],"after_bits":b[i],"before":val(a[i]),"after":val(b[i])} for i in range(env*48,(env+1)*48) if a[i]!=b[i]]
rows=collections.defaultdict(dict)
with open("instrumented.stdout",newline="") as f:
 for r in csv.DictReader(f,delimiter="\t"):rows[(r["arm"],int(r["step"]),int(r["env"]))][int(r["actor_component"])]=r
out["status"]={f"{arm}_{step}_{env}":{k:v for k,v in next(iter(x.values())).items() if k in ("status","physics_error","done")} for (arm,step,env),x in rows.items()}
out["unaffected_step3_actor_mismatch"]=[i for i in range(140) if rows[("failure",3,1)][i]["value"]!=rows[("control",3,1)][i]["value"]]
out["instrumented_equals_unobserved_text"]=open("instrumented.stdout","rb").read()==open("unobserved.stdout","rb").read()
print(json.dumps(out,indent=2))
