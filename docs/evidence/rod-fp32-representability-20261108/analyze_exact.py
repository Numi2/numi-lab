#!/usr/bin/env python3
"""Independent high-precision oracle for the frozen Metal input/output."""
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import struct

getcontext().prec=80
p=Path(__file__).resolve().parent
rows=json.loads((p/"inputs.json").read_text())
data=(p/"output-a.bin").read_bytes()
assert data==(p/"output-b.bin").read_bytes() and len(data)==12*len(rows)==504
D=Decimal.from_float
records=[]
for row,(naive,high,low) in zip(rows,struct.iter_unpack("<fff",data)):
    l,c,r,L,R,kl,kr,vo,vn,a=[D(hi)+D(lo) for hi,lo in row["pairs"]]
    expected=vn-vo+a*(kl*(c-l-L)-kr*(r-c-R))
    found=D(high)+D(low)
    err=abs(found-expected)
    assert err.is_finite()
    records.append({"label":row["label"],"node":row["node"],
                    "expected":str(expected),"found":str(found),"absolute_error":str(err)})
worst=max(Decimal(r["absolute_error"]) for r in records)
assert worst < Decimal("0.0001")
# Deliberately perturb a captured result well beyond the gate: the oracle
# must reject the resulting observation, rather than a coincidental all-pass.
first=records[0]
perturbed=abs(Decimal(first["found"])+Decimal("0.01")-Decimal(first["expected"]))
assert perturbed > Decimal("0.0001")
out={"precision_digits":getcontext().prec,"rows":len(records),
     "worst_absolute_error_m_per_s":str(worst),
     "perturbed_result_detected":True,
     "output_sha256":hashlib.sha256(data).hexdigest(),
     "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     "rows_detail":records}
(p/"analysis-exact.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="rows_detail"},indent=2))
