from pathlib import Path
import collections,csv,hashlib,json,statistics
W=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
p0=W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"
p1=W/"root-b109-alpha-0p25-audit-130-001/query/pairs.csv"
o=W/"root-b109-alpha-0p25-regression-001/inside-depth-comparison.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline="") as f:
  rows=list(csv.DictReader(f))
 out={(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"])):r for r in rows}
 assert len(out)==len(rows)
 return out
a,b=read(p0),read(p1)
assert set(a)==set(b)
groups=collections.defaultdict(list);new=[];cleared=[]
for k in sorted(a):
 x,y=a[k],b[k]
 if x["classification"]!="inside" and y["classification"]=="inside":new.append(k)
 if x["classification"]=="inside" and y["classification"]!="inside":cleared.append(k)
 if x["classification"]==y["classification"]=="inside":
  groups[(k[1],k[2])].append({"query_row":k[0],
    "before_m":float(x["nearest_distance_m"]),"after_m":float(y["nearest_distance_m"]),
    "delta_m":float(y["nearest_distance_m"])-float(x["nearest_distance_m"])})
summary=[]
for (body,bone),rows in sorted(groups.items()):
 vals=[r["delta_m"] for r in rows]
 summary.append({"body":body,"stable_bone_id":bone,"count":len(vals),"min_delta_m":min(vals),
                 "median_delta_m":statistics.median(vals),"max_delta_m":max(vals),"samples":rows})
o.write_text(json.dumps({"status":"POSITION_ONLY_MATCHED_QUERY_DEPTH_COMPARISON",
 "baseline_csv_sha256":sha(p0),"candidate_csv_sha256":sha(p1),"query_rows":len(a),
 "new_inside":new,"cleared_inside":cleared,"banks":summary,"physical_admission":False,
 "interpretation":"Negative deltas improve nearest-bone penetration depth, positive deltas worsen it; an unchanged inside classification is not a cleared point."},indent=2)+"\n")
print("wrote",o,"sha256",sha(o),"new_inside",len(new),"cleared_inside",len(cleared))
