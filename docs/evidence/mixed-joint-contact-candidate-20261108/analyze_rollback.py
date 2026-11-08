import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent/"rollback"
d={k:json.loads((p/(k+".stdout")).read_text()) for k in ("clean10","clean11","injected11")}
a,b,c=d["clean10"],d["injected11"],d["clean11"]
for name,x in d.items(): assert all(x[k] for k in ("capture_noninvasive","repeat","exact_trace")),name
assert (a["success"],a["failed"])==(10,0)
assert (b["success"],b["failed"])==(10,1)
assert (c["success"],c["failed"])==(11,0)
assert b["steps"][-1]["code"]==11 and b["steps"][-1]["unchanged_from_previous"]
assert a["steps"]==b["steps"][:10]
assert len(b["steps"][-1]["rows"])==3 and all(r["scratch"][3]>0 for r in b["steps"][-1]["rows"])
assert a["final_bytes"]==b["final_bytes"]
assert a["final_bytes"]!=c["final_bytes"]
for key,value in a["final_bytes"].items():
 assert value and len(value)%2==0
 bytes.fromhex(value)
result={"clean_accepted":a["success"],"injected_accepted":b["success"],"injected_rejected":b["failed"],
 "failure_status":b["steps"][-1]["code"],"joint_markers":[r["scratch"][3] for r in b["steps"][-1]["rows"]],
 "prefix_exact":True,"rejected_exposed_state_unchanged":True,"final_owners_byte_equal":True,
 "clean_eleven_negative_control_different":True,"owners":{key:{"bytes":len(value)//2,"sha256":hashlib.sha256(bytes.fromhex(value)).hexdigest()} for key,value in a["final_bytes"].items()}}
(p/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
