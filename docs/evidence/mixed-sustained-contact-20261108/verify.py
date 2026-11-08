import copy,json
from pathlib import Path

def load(n): return json.loads(Path(n).read_text())
def close(a,b,tol=1e-12): assert abs(a-b)<=tol,(a,b)
a=load("analysis.json"); one=load("one-ms.json"); ref=load("reference-refinement.json"); cap=load("cap-analysis.json")
for n in ("fixed64","bounded"):
 d=load(n+".json");assert (d["success"],d["failed"])==(20,0)
 assert d["capture_noninvasive"] and d["repeat"] and d["exact_trace"]
f=[s["ground_normal_impulse_Ns"]/.001 for s in a["fixed64"]["steps"]];g=[s["ground_normal_impulse_Ns"]/.001 for s in a["bounded"]["steps"]]
close(max(abs(x-y) for x,y in zip(f,g))/max(map(abs,f)),one["max_force_error_relative_peak"])
assert one["max_force_error_relative_peak"]>.0025 and not one["force_gate_0_25_percent_pass"]
assert ref["both_below_0_25_percent"] and ref["force_error_relative_128_peak"]<.0025
r=load("cap2-v2.json");assert (r["success"],r["failed"])==(0,20) and r["final_restored"] and all(s["step_restored"] for s in r["steps"])
for threshold,count in (("2e-4",8),("2e-5",7)):
 r=load("candidate-"+threshold+"-reporting.json");assert (r["success"],r["failed"])==(count,20-count)
 assert all(s["unchanged_from_previous"] for s in r["steps"][count:])
 longer=load("candidate-"+threshold+"-cap128.json");assert r["steps"][:count]==longer["steps"][:count]
assert cap["2e-4"]["success_count"]==20 and cap["2e-4"]["one_ms_force_error_relative_peak"]>.0025
assert cap["2e-5"]["success_count"]==7 and cap["2e-5"]["failed_count"]==13
x=load("rowtrace-control.json");y=load("rowtrace.json");old=load("candidate-2e-5-cap128.json")
def strip(d):
 d=copy.deepcopy(d)
 for s in d["steps"]:
  for row in s["rows"]: row.pop("row_trace")
 return d
assert strip(x)==old and strip(y)==strip(x)
s=y["steps"][7];assert s["code"]!=0 and s["unchanged_from_previous"]
rod=[r for r in s["rows"] if r["flags"]&8];rigid=[r for r in s["rows"] if not r["flags"]&8];assert len(rod)==2 and len(rigid)==1
close(max(abs(r["row_trace"][0]) for r in s["rows"]),s["normal_residual"],1e-10)
close(max(r["row_trace"][1] for r in s["rows"]),s["failure_tangent_diagnostic"],1e-10)
assert max(abs(r["row_trace"][0]) for r in rod)>1000*abs(rigid[0]["row_trace"][0])
print("Retained force gates, refinement, rejection, restoration observations and row attribution verified.")
