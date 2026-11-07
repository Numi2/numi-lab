import collections,csv,json,math,struct
groups=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
 for r in csv.DictReader(f,delimiter="\t"):
  groups[(r["arm"],int(r["phase"]),r["array"])][int(r["index"])]=int(r["bits"])
def val(x):return struct.unpack("<f",struct.pack("<I",x))[0]
out={}
for arm in ("control","failure"):
 for phase in (1,2,3):
  ids=groups[(arm,phase,"frameIdentity")];depth=groups[(arm,phase,"frameDepth")];valid=groups[(arm,phase,"frameValidity")];binding=groups[(arm,phase,"binding")]
  for env in (0,1):
   pixels=[]
   for p in range(env*144,(env+1)*144):
    identity=[ids[4*p+i] for i in range(4)]
    if identity[1]!=4294967295:
     pixels.append(dict(pixel=p-env*144,identity=identity,validity=valid[p],depth=val(depth[p])))
   out[f"{arm}_{phase}_{env}"]=dict(binding_ids=list(binding.values()),nonbackground_pixels=pixels)
out["actor_status_matches_prior_text"]=open("run.stdout","rb").read()==open("../g1-free-spheres-track-20261107/instrumented.stdout","rb").read()
print(json.dumps(out,indent=2))
