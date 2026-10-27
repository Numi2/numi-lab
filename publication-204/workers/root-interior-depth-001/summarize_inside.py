from pathlib import Path
import collections,csv,hashlib,json,statistics
W=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
Q=W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"
S=W/"combined-d071-b109-b61-001/audit-130/summary.json"
OUT=W/"root-interior-depth-001/inside-depths.json"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
summary=json.loads(S.read_text())
assert summary["candidate_sha256"]=="806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756"
groups=collections.defaultdict(list)
seen=set()
with Q.open(newline="") as f:
 for row in csv.DictReader(f):
  if row["classification"]!="inside":continue
  key=(int(row["bank_body"]),int(row["bank_stable_id"]))
  identity=(int(row["query_row"]),)+key
  assert identity not in seen
  seen.add(identity)
  d=float(row["nearest_distance_m"])
  assert d>=0
  groups[key].append({"query_row":identity[0],"nearest_distance_m":d,"nearest_bone_triangle":int(row["nearest_triangle"])})
assert len(seen)==125
results=[]
for (body,bone),rows in sorted(groups.items()):
 ds=[r["nearest_distance_m"] for r in rows]
 results.append({"body":body,"stable_bone_id":bone,"count":len(rows),
                 "minimum_distance_m":min(ds),"median_distance_m":statistics.median(ds),"maximum_distance_m":max(ds),
                 "samples":sorted(rows,key=lambda r:r["query_row"])})
OUT.write_text(json.dumps({"status":"OFFLINE_REFERENCE_POSE_QUERY_DEPTH_SUMMARY",
 "candidate_sha256":summary["candidate_sha256"],"summary_sha256":sha(S),"pairs_csv_sha256":sha(Q),
 "inside_node_bank_rows":len(seen),"banks":results,
 "interpretation":"Distances are point-to-nearest-bone-surface magnitudes for query rows classified inside. They are displacement lower bounds for clearance against the fixed bone, not proof of a feasible repair or physical/native qualification."},indent=2)+"\n")
print("wrote",OUT,"sha256",sha(OUT))
