from pathlib import Path
import csv, hashlib, json, math
from collections import Counter, defaultdict
root=Path(__file__).resolve().parent.parent
source=root/"crow-v15-corvid-flight-data-registry-20261112/rsfs20160081_si_001.csv"
data=list(csv.DictReader(source.open(newline="")))
rows=[]
for row in data:
    if row["MoultStage"]!="pre-moult": continue
    try:
        u=float(row["AirSpeed"])
        drag=sum(float(row[k]) for k in ("DragInduced","DragProfile","DragBody"))
    except ValueError: continue
    if u>0 and drag>0 and math.isfinite(drag): rows.append((u,drag,row["ID"][:12],row["ID"]))
train=[r for r in rows if not 9.4<=r[0]<=10.2]
test=[r for r in rows if 9.4<=r[0]<=10.2]
assert train and test and not ({x[2] for x in train}&{x[2] for x in test})
counts=Counter(x[2] for x in train)
s11=s12=s22=t1=t2=0.0
for u,d,g,_ in train:
    x1=u*u; x2=1/(u*u); w=1/counts[g]
    s11+=w*x1*x1; s12+=w*x1*x2; s22+=w*x2*x2; t1+=w*x1*d; t2+=w*x2*d
det=s11*s22-s12*s12
assert det>0
candidates=[((t1*s22-t2*s12)/det,(t2*s11-t1*s12)/det),(max(0,t1/s11),0),(0,max(0,t2/s22)),(0,0)]
def cost(c): return sum((c[0]*u*u+c[1]/(u*u)-d)**2/counts[g] for u,d,g,_ in train)
a,b=min((c for c in candidates if min(c)>=0),key=cost)
def score(rr):
    errors=[a*u*u+b/(u*u)-d for u,d,_,_ in rr]
    groups=defaultdict(list)
    for item,e in zip(rr,errors): groups[item[2]].append(e)
    ge=[sum(v)/len(v) for v in groups.values()]
    return {"records":len(rr),"provisional_groups":len(groups),"group_ids":sorted(groups),"airspeed_m_s":[min(r[0] for r in rr),max(r[0] for r in rr)],"record_mae_N":sum(map(abs,errors))/len(errors),"record_bias_N":sum(errors)/len(errors),"group_mean_error_mae_N":sum(map(abs,ge))/len(ge),"group_mean_error_bias_N":sum(ge)/len(ge)}
out={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"form":"D_N = a * (U_m_s)^2 + b / (U_m_s)^2","a":a,"b":b,"training":score(train),"heldout_9_4_to_10_2":score(test),"independent_birds":1,"model_scope":"one premoult jackdaw; steady glide only; no low-speed or powered flight validation","minimum_data_speed_m_s":min(r[0] for r in rows),"maximum_data_speed_m_s":max(r[0] for r in rows)}
p=Path(__file__).with_name("fit.json");p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
