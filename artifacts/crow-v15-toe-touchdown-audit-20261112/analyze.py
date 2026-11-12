from pathlib import Path
import collections,gzip,importlib.util,json,math
s=importlib.util.spec_from_file_location("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
A=Path("artifacts/crow-v15-toe-touchdown-audit-20261112");P=Path("artifacts/crow-v15-early-toe-release-screen-20261112");Q=Path("artifacts/crow-v15-short-toe-swing-screen-20261112");paths={"sham":P/"sham-20261126-paired","early":P/"candidate-20261126-rebound","short":Q/"candidate-20261126-rebound"};out={};threshold=.5*.45006*9.81*.02/16
for name,path in paths.items():
 gait=a.analyze(path,0.0);q={};v={};n={};targets={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   z=l.split();key=z[0]
   if key in ("CROW_Q_SUBSTEP","CROW_TOE_V_SUBSTEP","CROW_TWO_TOE_SUBSTEP"):
    t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     if key=="CROW_Q_SUBSTEP":q[e,t,k]=list(map(float,z[4:]))
     elif key=="CROW_TOE_V_SUBSTEP":v[e,t,k]=list(map(float,z[4:]))
     else:n[e,t,k]=[float(z[8]),float(z[9])]
   elif key=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3])
    if 201<=t<=300:targets[e,t]=list(map(float,z[3:]))
 assert len(q)==len(v)==len(n)==8*100*16 and len(targets)==800
 events=[]
 for e in range(8):
  for side in range(2):
   for bout in gait[e]["support_qualified_swing_bouts"][side]:
    end=bout[1];start=(end-201+1)*16;stop=min(100*16,start+5*16);found=None
    for i in range(start,stop-2):
     def key(j):return (e,201+j//16,j%16)
     if all(n[key(i+j)][side]>=threshold for j in range(3)):
      t=201+i//16;k=i%16;x=key(i);found={"interval":t,"substep":k,"delay_ms":(i-start+1)*1.25,"proximal_toe_position_rad":q[x][29 if side==0 else 37],"proximal_toe_velocity_rad_s":v[x][6 if side==0 else 14],"proximal_target_rad":targets[e,t][28 if side==0 else 36],"foot_normal_impulse_Ns":n[x][side]};break
    events.append({"lane":e,"side":"left" if side==0 else "right","clearance_bout":bout,"loaded_contact":found})
 out[name]={"events":events,"missing":sum(x["loaded_contact"] is None for x in events)}
result={"half_weight_substep_threshold_Ns":threshold,"arms":out,"scope":"Heuristic first 3-consecutive-substep half-weight foot normal impulse following each endpoint-defined full-foot clearance bout, within 100 ms. This is an instrumented event proxy, not ground-truth anatomical touchdown. Missing events remain missing. One previously used development seed; no applied torque or biological measurement."};(A/"analysis.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({n:{"events":len(v["events"]),"missing":v["missing"]} for n,v in out.items()},indent=2))
