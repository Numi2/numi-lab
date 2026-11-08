import json, math
from pathlib import Path
p=Path(__file__).resolve().parent

def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def rotate(q,r):
    u=q[3:6]; w=q[6]; c=cross(u,r); d=dot(u,r); uu=dot(u,u)
    return [(w*w-uu)*r[i]+2*u[i]*d+2*w*c[i] for i in range(3)]
def metric(r,v,row):
    w=v[3:6];c=cross(w,r);point_v=[v[i]+c[i] for i in range(3)]
    slip=[dot(row["directions"][k],point_v)-row["tangent_targets"][k-1] for k in (1,2)]
    speed=math.hypot(*slip); normal=row["impulse"]; pt=row["tangent_impulses"]; limit=.5*normal; tolerance=1e-5+.001*limit
    err=math.hypot(pt[0]+limit*slip[0]/speed,pt[1]+limit*slip[1]/speed) if speed>1e-5 else max(math.hypot(*pt)-limit,0.)
    return {"speed_mps":speed,"error_ns":err,"tolerance_ns":tolerance,"pass":err<=tolerance}
out={}
for arm in ("warm","cold"):
    data=json.loads((p/("friction-"+arm+".json")).read_text()); failures=[];disagree=[];max_delta=0.;checked=0;maximum=None
    for ci,case in enumerate(data["cases"]):
        for si,step in enumerate(case["steps"]):
            if not step["accepted"]:continue
            prev=case["steps"][si-1]["q"] if si else [0,.01+case["initial_gap"],0,0,0,0,1]
            for ri,row in enumerate(step["rows"]):
                assert row["body_a"]==0 and row["body_b"]==1
                local=rotate(prev,row["anchor_b"]);world=[row["point"][i]-prev[i] for i in range(3)]
                delta=math.sqrt(sum((local[i]-world[i])**2 for i in range(3)))
                if delta>max_delta:
                    max_delta=delta;maximum={"case":ci,"step":si+1,"row":ri,"local_m":local,"point_m":world,"delta_m":delta}
                a=metric(local,step["v"],row);b=metric(world,step["v"],row)
                record={"case":ci,"step":si+1,"row":ri,"local":a,"original":b,"lever_delta_m":delta}
                if not a["pass"]:failures.append(record)
                if a["pass"]!=b["pass"]:disagree.append(record)
                checked+=1
    out[arm]={"rows":checked,"max_lever_delta_m":max_delta,"max_location":maximum,"local_failure_count":len(failures),"gate_disagreement_count":len(disagree),"failures":failures,"disagreements":disagree}
(p/"lever-audit.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:{a:b for a,b in v.items() if a not in ("failures","disagreements")} for k,v in out.items()},indent=2))
