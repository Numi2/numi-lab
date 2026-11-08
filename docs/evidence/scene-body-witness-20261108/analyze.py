import json, math
from pathlib import Path
p=Path(__file__).resolve().parent
raw=json.loads((p/"output.json").read_text());out=[]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
for c in raw["cases"]:
    assert c["accepted"] and c["replay"] and c["contacts"]==1 and len(c["rows"])==1
    r=c["rows"][0];assert r["a"]==0 and r["b"]==c["global_sphere"]==2
    n=r["normal"][:3];t=r["tangent"][:3];b=cross(n,t);j=r["impulse"]
    impulse=[n[i]*j[0]+t[i]*j[1]+b[i]*j[2] for i in range(3)]
    local=r["anchor_b"][:3];mid=[r["point"][i]-c["start_position"][i] for i in range(3)]
    anchor_w=[1000*x for x in cross(local,impulse)];mid_w=[1000*x for x in cross(mid,impulse)];actual=c["end_angular"][:3]
    err=lambda x:max(abs(x[i]-actual[i]) for i in range(3))
    mom=max(abs(c["end_velocity"][i]-c["start_velocity"][i]-impulse[i]) for i in range(3))
    expected_gap=c["gap"];measured_gap=r["point"][3]
    assert abs(expected_gap-measured_gap)<1e-7
    out.append({"gap_m":expected_gap,"reported_gap_m":measured_gap,"impulse_Ns":impulse,"actual_delta_omega_rad_s":actual,
      "local_witness_delta_omega_rad_s":anchor_w,"midpoint_delta_omega_rad_s":mid_w,"local_error_rad_s":err(anchor_w),"midpoint_error_rad_s":err(mid_w),"momentum_error_Ns":mom,
      "discriminating":max(abs(anchor_w[i]-mid_w[i]) for i in range(3))>=1e-3})
assert len(out)==4 and all(x["momentum_error_Ns"]<1e-6 for x in out)
assert out[0]["local_error_rad_s"]<1e-4 and out[0]["midpoint_error_rad_s"]<1e-4
assert all(x["discriminating"] and x["local_error_rad_s"]>1e-4 and x["midpoint_error_rad_s"]<1e-4 for x in out[1:3])
assert max(abs(v) for v in out[3]["impulse_Ns"])==0 and max(abs(v) for v in out[3]["actual_delta_omega_rad_s"])==0
(p/"analysis.json").write_text(json.dumps({"cases":out},indent=2)+"\n")
print(json.dumps({"cases":out},indent=2))
