import collections,csv,json,struct
groups=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
 for row in csv.DictReader(f,delimiter="\t"):
  groups[(row["arm"],int(row["phase"]),row["array"])][int(row["index"])]=int(row["bits"])
def val(bits):return struct.unpack("<f",struct.pack("<I",bits))[0]
out={}
for arm in ("control","failure"):
 for phase in range(4):
  track=groups[(arm,phase,"track")]
  for env in (0,1):
   out[f"{arm}_{phase}_{env}_confidence"]=[val(track[env*48+i*8+3]) for i in range(6)]
for env in (0,1):
 a=groups[("failure",2,"track")];b=groups[("failure",3,"track")]
 out[f"failure_env_{env}_changed_track_bits"]=[i for i in range(env*48,(env+1)*48) if a[i]!=b[i]]
out["actor_status_exact_text_equal"]=open("instrumented.stdout","rb").read()==open("unobserved.stdout","rb").read()
print(json.dumps(out,indent=2))
