from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-tiny-toe-divergence-audit-20261112");B=Path("artifacts/crow-v15-tiny-swing-toe-screen-20261112");rows=[]
for arm,d in (("sham","sham-20261127-dev2"),("candidate","candidate-20261127-rebound")):
 target={};v={}
 with gzip.open(B/d/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_DRIVE_TARGET "):
    z=line.split();t,e=map(int,z[1:3]);assert len(z)==43
    if 210<=t<=230:target[e,t]=list(map(float,z[3:]))
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);assert len(z)==20
    if 210<=t<=230:v[e,t,k]=list(map(float,z[4:]))
 assert len(target)==8*21 and len(v)==8*21*16
 for e in range(8):
  for t in range(210,231):
   values=[v[e,t,k][14] for k in range(16)];rows.append({"arm":arm,"lane":e,"interval":t,"right_rear_proximal_command_rad":target[e,t][36],"right_rear_proximal_velocity_min_rad_s":min(values),"right_rear_proximal_velocity_max_rad_s":max(values)})
out={"rows":rows,"scope":"Recorded position target and post-substep joint velocity, not applied torque. Both sampled from the original contact-switched run; target is held in each control interval. Results alone do not prove the travel-loss mechanism."};(A/"drive.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps([r for r in rows if r["lane"]==0 and r["interval"] in (214,215,216,220,225)],indent=2))
