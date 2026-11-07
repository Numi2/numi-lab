import collections,csv,json,struct
group=collections.defaultdict(dict)
with open("snapshot.tsv",newline="") as f:
    for row in csv.DictReader(f,delimiter="\t"):
        group[(row["arm"],int(row["phase"]),row["array"])][int(row["index"])]=int(row["bits"])
summary={}
for arm in ("control","failure"):
    for env in (0,1):
        before=group[(arm,0,"depth")]
        after=group[(arm,1,"depth")]
        first=env*19*16*9
        last=(env+1)*19*16*9
        assert all(i in before and i in after for i in range(first,last))
        changes=[i for i in range(first,last) if before[i]!=after[i]]
        values=collections.Counter(struct.unpack("<f",struct.pack("<I",after[i]))[0] for i in range(first,last))
        summary[f"{arm}_env_{env}"]={"depth_values_changed":len(changes),"after_value_counts":dict(sorted(values.items()))}
summary["actor_status_streams_exact_text_equal"]=open("instrumented.stdout","rb").read()==open("unobserved.stdout","rb").read()
print(json.dumps(summary,indent=2))
