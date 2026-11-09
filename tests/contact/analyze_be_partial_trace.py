#!/usr/bin/env python3
"""Read an immutable 5 ms prefix of a running BE trace; this does not qualify a rollout."""
import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
source=p/"be-precision-005-n2-h00003125.json"
reference=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
output=p/"be-fine-5ms-partial-diagnostic.json"
if output.exists(): raise SystemExit("diagnostic already exists")
data=source.read_bytes()
marker=b'"traces":['
start=data.find(marker)
if start<0: raise SystemExit("trace has not started")
header=json.loads(data[:start]+b'"traces":[]}')
h=header["h"];steps=160000
if header["arm"]!="ipc_area" or header["divisions"]!=2 or h!=3.125e-8: raise SystemExit("unexpected fixture")
text=data.decode("utf-8");offset=start+len(marker);decoder=json.JSONDecoder()
bins=[0.]*5;impulse=0.;peak=0.;clearance=math.inf;rejected=0;contacts=0
for i in range(steps):
 while offset<len(text) and text[offset] in " ,\n\r\t": offset+=1
 try: row,offset=decoder.raw_decode(text,offset)
 except json.JSONDecodeError: raise SystemExit("5 ms prefix is not complete")
 if row["step"]!=i: raise SystemExit("noncontiguous steps")
 if not row["accepted"]: rejected+=1
 else:
  impulse+=row["accepted_normal_impulse"]
  bins[min(4,int((i+.5)*h/.001))]+=row["accepted_normal_impulse"]/.001
  peak=max(peak,row["force"]);clearance=min(clearance,row["gap"])
  contacts+=row["contacts"]>0
r=json.loads(reference.read_text())
if r["mesh_divisions"]!=2: raise SystemExit("reference mesh mismatch")
refrow=next(q for q in r["trace"] if abs(q[0]-.005)<1e-12)
result={"schema":"numi.be.partial-contact-diagnostic.v1",
 "scope":"first 5 ms only; no complete-rollout, replay, accuracy qualification, or speed claim",
 "h_s":h,"accepted_rows":steps-rejected,"rejected_rows":rejected,"contact_rows":contacts,
 "min_sampled_gap_m":clearance,"binned_force_N":bins,"reference_binned_force_N":r["force_bins_N"][:5],
 "max_binned_force_difference_over_full_reference_peak":max(abs(x-y) for x,y in zip(bins,r["force_bins_N"][:5]))/max(r["force_bins_N"]),
 "partial_impulse_Ns":impulse,"reference_partial_impulse_Ns":refrow[2],
 "partial_impulse_difference_over_full_reference_impulse":abs(impulse-refrow[2])/r["impulse_Ns"],
 "partial_sampled_peak_force_N":peak,"reference_full_sampled_peak_force_N":r["peak_force_N"],
 "source_prefix_bytes":offset,"source_prefix_sha256":hashlib.sha256(data[:offset]).hexdigest(),
 "reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest()}
output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,indent=2,sort_keys=True))

