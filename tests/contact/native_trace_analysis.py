#!/usr/bin/env python3
"""Compare accepted native force impulses to one temporally qualified mesh.

The native trace is streamed twice: first for complete metadata, then for the
same ordered arithmetic as the original analyzer. Only the cumulative entries
needed by the reference clock are retained.
"""
import json,math,sys
from streamed_native_trace import read_native

def load_reference(path):
 rows=[]
 metadata,count=read_native(path,lambda _meta,row,_index:rows.append(row),trace_key="trace")
 assert count==len(rows)
 metadata["trace"]=rows
 return metadata
a,count=read_native(sys.argv[1]);r=load_reference(sys.argv[2])
assert a["divisions"]==r["mesh_divisions"]
steps=round(.01/a["h"])
complete=a["accepted_steps"]==steps and a["rejected_steps"]==0 and count==steps
out={"complete":complete,"mesh":a["divisions"],"arm":a["arm"],"h":a["h"]}
weights=a["contact_area_weights_m2"];rw=r["contact_area_weights_m2"]
assert len(weights)==len(rw)
out["area_sum_m2"]=sum(weights)
out["area_sum_abs_error_m2"]=abs(sum(weights)-.0004)
out["maximum_weight_error_normalized_area"]=max(abs(x-y) for x,y in zip(weights,rw))/.0004
out["area_valid"]=out["area_sum_abs_error_m2"]<=8e-11 and out["maximum_weight_error_normalized_area"]<=2e-7
if complete:
 h=a["h"];assert len(r["trace"])==5001
 bins=[0.]*10
 needed=set()
 for t,_,_ in r["trace"]:
  k=min(int(t/h),steps-1);needed.update((k,k+1))
 cumulative={0:0.};state={"sum":0.,"peak":-math.inf}
 def accept(meta,q,index):
  j=q["step"];assert q["accepted"] and j==index
  bins[min(9,int((j+.5)*h/.001))]+=q["force"]*h/.001
  state["sum"]+=q["accepted_normal_impulse"]
  state["peak"]=max(state["peak"],q["force"])
  if index+1 in needed:cumulative[index+1]=state["sum"]
 meta2,count2=read_native(sys.argv[1],accept)
 assert count2==count and meta2==a
 def interp(t):
  k=min(int(t/h),steps-1)
  f=max(0.,min(1.,(t-k*h)/h))
  return cumulative[k]+f*(cumulative[k+1]-cumulative[k])
 peak=state["peak"];refpeak=r["peak_force_N"];refimp=r["impulse_Ns"]
 assert refpeak>0 and refimp>0 and max(r["force_bins_N"])>0
 out.update({"binned_force_error":max(abs(x-y) for x,y in zip(bins,r["force_bins_N"]))/max(r["force_bins_N"]),
  "total_impulse_error":abs(state["sum"]-refimp)/refimp,
  "peak_force_error":abs(peak-refpeak)/refpeak,
  "cumulative_impulse_error":max(abs(interp(t)-imp) for t,_,imp in r["trace"])/refimp,
  "native_peak_force_N":peak,"reference_peak_force_N":refpeak,
  "native_impulse_Ns":state["sum"],"reference_impulse_Ns":refimp,
  "peak_sampling_seconds":h,"cumulative_reference_sampling_seconds":2e-6})
 out["physical_valid"]=a["minimum_gap"]>1e-7 and a["law_max_relative_error"]<=1e-4 and a["independent_max_local_scaled_residual"]<=.005 and a["momentum_max_absolute_error_Ns"]<=1e-7 and a["restore_replay"] and a["forced_rollback_exact"]
 out["passes"]=out["area_valid"] and out["physical_valid"] and out["binned_force_error"]<=.0025 and out["total_impulse_error"]<=.0025 and out["peak_force_error"]<=.01 and out["cumulative_impulse_error"]<=.01
else:out["physical_valid"]=False;out["passes"]=False
print(json.dumps(out,indent=2,sort_keys=True))
sys.exit(0 if out["passes"] else 1)
