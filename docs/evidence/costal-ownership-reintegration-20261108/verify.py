import json
from pathlib import Path

def read(n):return Path(n).read_text()
def load(n):return json.loads(read(n))
assert "100% tests passed out of 3" in read("ctest-final.log")
assert "25% tests passed, 3 tests failed out of 4" in read("existing-ctest.log")
assert "100% tests passed out of 4" in read("existing-repair-ctest.log")
a=load("v5-asset-hashes.json");assert len(a)==9 and all(v["expected"]==v["actual"] for v in a.values())
r={x["name"]:x for x in load("native-final.json")};assert len(r)==11 and all(x.get("exit")==0 for x in r.values())
assert "body_contact=0" in r["adapter-empty"]["stdout"] and "body_contact=1" in r["adapter-contact"]["stdout"]
assert "closed=1 accepted=1" in r["adapter-contact"]["stderr"]
for n in ("owner-bounded","owner-rollback","owner-rollback-repeat"):
 s=r[n]["stdout"];assert s.count("PASS accepted costal owner step=")==4 and "Matter_authority_restored=1 HumanIO_rejected=1" in s
assert "root/q/v/mujoco=48,644,640,16" in r["owner-rollback"]["stdout"]
assert r["owner-rollback"]["stdout"]==r["owner-rollback-repeat"]["stdout"] and r["owner-rollback"]["stderr"]==r["owner-rollback-repeat"]["stderr"]
assert "Matter authority and forces bitwise" in r["owner-replay"]["stdout"]
for n in ("v5-admission","v5-admission-repeat"):
 s=r[n]["stdout"];assert "rejects mismatched binding identity status=3" in s and "source_fingerprint=8757514333639552933" in s
assert r["v5-admission"]["stdout"]==r["v5-admission-repeat"]["stdout"]
m=load("owner-mismatch.json");assert m["changed_offset"]==260 and m["original_owner"]==20 and m["altered_owner"]==25
assert all(x["exit"]==1 and "costal attachment admission failed" in x["stderr"] for x in m["runs"])
assert m["runs"][0]["stdout"]==m["runs"][1]["stdout"] and m["runs"][0]["stderr"]==m["runs"][1]["stderr"]
assert "AssertionError" in read("owner-mismatch-initial.log")
print("Combined binding, native ownership, rejection, replay and existing regression evidence verified.")
import re
assert "100% tests passed out of 3" in read("main-ctest-cpu.log")
assert "100% tests passed out of 4" in read("main-ctest-native.log")
main={x["name"]:x for x in load("main-native.json")};assert set(main)==set(r) and all(x.get("exit")==0 for x in main.values())
for n in main:
 a=re.sub(r"token=\d+","token=IDENTITY",r[n]["stdout"]);b=re.sub(r"token=\d+","token=IDENTITY",main[n]["stdout"]);assert a==b,n
assert main["owner-rollback"]["stdout"]==main["owner-rollback-repeat"]["stdout"] and main["owner-rollback"]["stderr"]==main["owner-rollback-repeat"]["stderr"]
m=load("main-owner-mismatch.json");assert all(x["exit"]==1 and "costal attachment admission failed" in x["stderr"] for x in m["runs"])
assert m["runs"][0]["stdout"]==m["runs"][1]["stdout"] and m["runs"][0]["stderr"]==m["runs"][1]["stderr"]
print("Fresh main build, native regressions and cross-build displayed results verified.")
