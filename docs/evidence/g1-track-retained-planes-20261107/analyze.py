import collections,csv,json,struct
groups=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
    for row in csv.DictReader(f,delimiter="\t"):
        groups[(row["arm"],int(row["phase"]),row["array"])][int(row["index"])]=int(row["bits"])
def val(bits): return struct.unpack("<f",struct.pack("<I",bits))[0]
out={}
for arm in ("control","failure"):
    for phase in range(4):
        a=groups[(arm,phase,"track")]
        for env in (0,1):
            out[f"{arm}_{phase}_{env}"]=[val(a[env*48+track*8+3]) for track in range(6)]
out["actor_status_streams_exact_text_equal"]=open("instrumented.stdout","rb").read()==open("unobserved.stdout","rb").read()
print(json.dumps(out,indent=2))
