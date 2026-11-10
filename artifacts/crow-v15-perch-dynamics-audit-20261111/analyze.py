from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-perch-dynamics-audit-20261111")
B=Path("artifacts/crow-v15-lean-20second-perch-screen-20261110")
def lines(p):
 with gzip.open(p,"rt") as f:
  for x in f:
   yield x.split()
old={int(v[3]):v[4:] for v in lines(B/"results.txt.gz") if v and v[0]=="trace"}
trace={};com={};w={};j={}
for v in lines(A/"results.txt.gz"):
 if not v:continue
 if v[0]=="trace":trace[int(v[3])]=v[4:]
 elif v[0]=="wrench":w[int(v[1])]=list(map(float,v[2:]))
 elif v[0]=="com":com[int(v[1])]=list(map(float,v[2:]))
 elif v[0]=="jointstate":j[(int(v[1]),int(v[2]))]=list(map(float,v[3:]))
diff=[t for t in sorted(set(old)|set(trace)) if old.get(t)!=trace.get(t)]
assert len(old)==len(trace)==len(com)==len(w)==1000 and len(j)==4000 and not diff,(len(old),len(trace),len(com),len(w),len(j),diff[:10])
rows=[]
for t in [0,100,150,200,250,300,350,400,450,500,550,600,650,700,800,900,999]:
 c=com[t];v=trace[t];p=2*(float(v[10]));pitch=math.asin(max(-1.,min(1.,p)))
 rows.append(dict(step=t,com_x=c[1],com_z=c[3],momentum_x=c[4],momentum_z=c[6],pitch_approx_rad=pitch,impulse=w[t],joints={str(k):j[t,k] for k in (14,16,17,19)}))
res=[]
for t in range(1,1000):
 c=com[t];prev=com[t-1];wj=w[t];res.append([c[4]-prev[4]-wj[0],c[6]-prev[6]-wj[2]+c[0]*9.81*.02])
summary={"trace_parity":True,"samples":len(trace),"mass_kg":com[0][0],"mean_absolute_momentum_residual_x_z":[sum(abs(r[k]) for r in res)/len(res) for k in range(2)],"selected":rows}
(A/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
