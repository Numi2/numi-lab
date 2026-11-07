#!/usr/bin/env python3
"""Independent time-refinement gate; never infer force accuracy from convergence."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("runs",nargs="+",type=Path)
a=p.parse_args()
runs=sorted([(path,json.loads(path.read_text())) for path in a.runs],key=lambda z:-z[1]["h"])
assert len(runs)>=2
assert len({x["divisions"] for _,x in runs})==1,"hold mesh fixed for temporal refinement"
assert all(x["arm"]=="ipc" for _,x in runs)
def bins(x):
    result=[0.]*10
    for t in x["traces"]:
        if t["accepted"]:
            i=int((t["step"]+.5)*x["h"]/.001)
            assert 0<=i<10,"this instrument compares a declared 10 ms duration"
            result[i]+=t["accepted_normal_impulse"]/.001
    return result
rows=[]
for path,x in runs:
    rows.append({"input":str(path),"h":x["h"],"complete":abs(x["accepted_simulated_seconds"]-.01)<1e-10 and x["rejected_steps"]==0,
       "local_gate":x["independent_max_local_scaled_residual"]<=.005,
       "clearance_gate":x["minimum_gap"]>1e-7,"balance_gate":x["momentum_max_absolute_error_Ns"]<=1e-7,
       "transaction_gate":x["restore_replay"] and x["forced_rollback_exact"] and x["identity_mismatch_rejected"],
       "force_bins_N":bins(x),"total_impulse_Ns":sum(t["accepted_normal_impulse"] for t in x["traces"] if t["accepted"])})
pairs=[]
for c,f in zip(rows,rows[1:]):
    force=max(abs(u-v) for u,v in zip(c["force_bins_N"],f["force_bins_N"]))/max(max(map(abs,f["force_bins_N"])),1e-30)
    impulse=abs(c["total_impulse_Ns"]-f["total_impulse_Ns"])/max(abs(f["total_impulse_Ns"]),1e-30)
    pairs.append({"coarse_h":c["h"],"fine_h":f["h"],"binned_force_relative_change":force,"integrated_impulse_relative_change":impulse})
complete=all(all(row[k] for k in ["complete","local_gate","clearance_gate","balance_gate","transaction_gate"]) for row in rows)
qualified=complete and pairs[-1]["binned_force_relative_change"]<=.0025 and pairs[-1]["integrated_impulse_relative_change"]<=.0025
print(json.dumps({"schema":"numi.contact.temporal-refinement.v1","mesh_divisions":runs[0][1]["divisions"],
 "force_bin_width_s":.001,"reference_gate":.0025,"control_force_gate":.01,"runs":rows,"pairs":pairs,
 "completed_residual_and_transaction_gates":complete,"temporal_reference_qualified":qualified,
 "force_qualified_control":False,"spatial_reference_qualified":False,
 "limits":["A qualified temporal reference is necessary but not sufficient: separate mesh refinement and the 1% control force gate are still required.",
 "The frozen IPC implementation itself chooses stiffness using h and nodal inverse response. This series tests its discrete limit, not fixed contact regularization.",
 "Native timings are retained as observations only. These correctness runs include readback/replay and possible diagnostic GPU overlap; no speed comparison is qualified."]},indent=2))
raise SystemExit(0 if qualified else 1)
