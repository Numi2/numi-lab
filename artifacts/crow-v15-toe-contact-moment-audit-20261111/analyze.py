from pathlib import Path
import gzip,json,statistics
A=Path("artifacts/crow-v15-toe-contact-moment-audit-20261111");B=Path("artifacts/crow-v15-aft-com-grip-grid-20261111/results.txt.gz")
def read(p,lane=1):
 tr={};g={};cx={};cz={}
 with gzip.open(p,"rt") as f:
  for l in f:
   v=l.split()
   if not v:continue
   if v[0]=="trace" and int(v[1])==lane and int(v[3])<250:tr[int(v[3])]=v[4:]
   elif v[0]=="groups":g[int(v[1])]=list(map(float,v[2:]))
   elif v[0]=="com_x":cx[int(v[3])]=float(v[4])
   elif v[0]=="com_z":cz[int(v[1])]=float(v[2])
 return tr,g,cx,cz
out={"runs":{}}
for n,lane in ((0,1),(1,7)):
 tr,g,cx,cz=read(A/f"trial-{n}.txt.gz");ref=read(B,lane)[0];diff=[t for t in range(250) if tr.get(t)!=ref.get(t)];assert len(tr)==len(g)==len(cx)==len(cz)==250 and not diff,(n,len(tr),len(g),len(cx),len(cz),diff[:20]);windows=[]
 for lo,hi in ((80,110),(110,150),(150,200),(200,240)):
  groups=[]
  for k,name in enumerate(("rear_toes","front_toes","feet","other")):
   r=[(g[t][3*k],g[t][3*k+1],g[t][3*k+2]-cz[t]*g[t][3*k]+cx[t]*g[t][3*k+1]) for t in range(lo,hi)]
   groups.append({"group":name,"mean_Jx":statistics.mean(x[0] for x in r),"mean_Jz":statistics.mean(x[1] for x in r),"mean_angular_impulse_about_endpoint_com":statistics.mean(x[2] for x in r)})
  windows.append({"ticks":[lo,hi-1],"groups":groups})
 out["runs"][str(n)]={"batch_reference_lane":lane,"exact_trace_parity":True,"windows":windows}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
