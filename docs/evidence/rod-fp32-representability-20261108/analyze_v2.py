#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path
import struct

p=Path(__file__).resolve().parent
rows=json.loads((p/"inputs.json").read_text())
raw=(p/"output-a.bin").read_bytes()
assert raw==(p/"output-b.bin").read_bytes() and len(raw)==12*len(rows)==504
def calculate(v):
    left,center,right,L,R,kl,kr,vold,vnew,alpha=v
    return vnew-vold+alpha*(kl*(center-left-L)-kr*(right-center-R))
out=[]
for row,(naive,high,low) in zip(rows,struct.iter_unpack("<fff",raw)):
    values=[a+b for a,b in row["pairs"]]
    reference=calculate(values)
    compensated=high+low
    assert all(math.isfinite(v) for v in (reference,naive,compensated))
    out.append({"label":row["label"],"node":row["node"],"reference":reference,
                "naive":naive,"compensated":compensated,
                "compensated_abs_error":abs(compensated-reference)})
retained=[x for x in out if x["label"]=="retained-offset-0.25"]
fresh=[x for x in out if x["label"]!="retained-offset-0.25"]
summary={"rows":len(out),"retained_rows":len(retained),"fresh_rows":len(fresh),
         "gpu_bitwise_replay":True,
         "max_reference_residual_m_per_s":max(abs(x["reference"]) for x in out),
         "max_naive_residual_m_per_s":max(abs(x["naive"]) for x in out),
         "max_compensated_residual_m_per_s":max(abs(x["compensated"]) for x in out),
         "max_compensated_error_m_per_s":max(x["compensated_abs_error"] for x in out),
         "naive_rows_above_gate":sum(abs(x["naive"])>.001 for x in out),
         "compensated_rows_above_gate":sum(abs(x["compensated"])>.001 for x in out),
         "retained_naive_rows_above_gate":sum(abs(x["naive"])>.001 for x in retained),
         "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
summary["prediction_passed"]=(summary["max_compensated_error_m_per_s"]<=.0001 and
                              summary["compensated_rows_above_gate"]==0 and
                              summary["retained_naive_rows_above_gate"]>0)
(p/"analysis-v2.json").write_text(json.dumps({"summary":summary,"rows":out},indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
