import collections,csv,json
group=collections.defaultdict(list)
with open("run.stdout",newline="") as stream:
    for row in csv.DictReader(stream,delimiter="\t"):
        group[(row["arm"],int(row["step"]),int(row["env"]))].append(row)
out={}
out["rows"]={"%s_%d_%d"%key:len(rows) for key,rows in sorted(group.items())}
out["statuses"]={"%s_%d_%d"%key:dict(status=rows[0]["status"],physics_error=rows[0]["physics_error"],done=rows[0]["done"]) for key,rows in sorted(group.items())}
left=group[("control",1,1)]
right=group[("failure",1,1)]
out["unaffected_environment_mismatches"]=[dict(component=a["actor_component"],control=a["value"],failure=b["value"]) for a,b in zip(left,right) if a["actor_component"]!=b["actor_component"] or a["value"]!=b["value"]]
print(json.dumps(out,indent=2))
