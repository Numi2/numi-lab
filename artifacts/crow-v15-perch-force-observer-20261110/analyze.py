from pathlib import Path
import gzip,json,subprocess,math
R=Path.cwd();A=R/"artifacts/crow-v15-perch-force-observer-20261110";O=R/"artifacts/crow-v15-landing-toe-ramp-20261110";name="diagnostic-reserved-final-j"
def samples(path):
 d={}
 with gzip.open(path,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);d[x["step"]]=x
 return d
x=samples(O/(name+".stderr.gz"));y=samples(A/(name+".stderr.gz"));common=sorted(set(x)&set(y));mismatch=[t for t in common if x[t]["q"]!=y[t]["q"] or x[t]["v"]!=y[t]["v"]];assert not mismatch and len(common)>2500,(len(common),mismatch[:5]);rows={t:z for t,z in y.items() if 2400<=t<=2713};inp="".join(str(t)+" "+" ".join(str(v) for v in z["q"])+"\n" for t,z in rows.items());p=subprocess.run([str(R/"build/crow-v15-perch-force-observer-20261110/com_probe")],input=inp,text=True,capture_output=True,check=True);com={int(v[0]):[float(a) for a in v[2:]] for v in (l.split() for l in p.stdout.splitlines())};result=[]
for t,z in rows.items():
 center=com[t];imp=[0.,0.,0.];tau=[0.,0.,0.];up=0.;weighted=0.
 for c in z["contacts"]:
  a,b=c["a_name"],c["b_name"]
  if "crow_course_perch" not in (a,b):continue
  bird=a if b=="crow_course_perch" else b;sgn=-1 if bird==a else 1
  if "_foot" not in bird and "_toe" not in bird:continue
  force=[sgn*sum(c[k][j]*c["impulses"][i] for i,k in enumerate(("normal","tangent_u","tangent_v"))) for j in range(3)];r=[c["point"][j]-center[j] for j in range(3)]
  torque=[r[1]*force[2]-r[2]*force[1],r[2]*force[0]-r[0]*force[2],r[0]*force[1]-r[1]*force[0]]
  for j in range(3):imp[j]+=force[j];tau[j]+=torque[j]
  w=max(0,force[2]);up+=w;weighted+=w*c["point"][0]
 if up>0:result.append({"step":t,"com":center,"vertical_impulse_Ns":up,"support_x":weighted/up,"support_minus_com_x":weighted/up-center[0],"impulse_Ns":imp,"angular_impulse_Nms":tau,"pitch_rad":math.asin(max(-1,min(1,2*(z["q"][6]*z["q"][4]-z["q"][5]*z["q"][3])))),"pitch_rate_rad_s":z["v"][4]})
(A/"force-analysis.json").write_text(json.dumps({"parity_common":len(common),"q_v_mismatches":mismatch,"sampled_perch_contact":result,"scope":"final-substep contact impulses, endpoint COM; not continuous dynamics"},indent=2)+"\n")
for t in (2500,2525,2550,2575,2600,2625,2650,2675,2700):
 r=next((r for r in result if r["step"]==t),None)
 if r:print(t,"pitch",round(r["pitch_rad"],3),"rate",round(r["pitch_rate_rad_s"],3),"support-COM x",round(r["support_minus_com_x"],4),"impulse",[round(a,5) for a in r["impulse_Ns"]],"pitch impulse",round(r["angular_impulse_Nms"][1],6))
print("parity",len(common),"mismatches",len(mismatch))
