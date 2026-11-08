import json,math
from pathlib import Path
root=Path(__file__).resolve().parent
runs=root/"runs";p=root/"witness"
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ("scratch","row_trace")}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
out={}
for n in (5,9,17):
 data=json.loads((p/("rod%d.stdout"%n)).read_text());base=json.loads((runs/("rod%d-long.stdout"%n)).read_text())
 assert exposed(data)==exposed(base) and all(data[k] for k in ("capture_noninvasive","repeat","exact_trace"))
 patterns={};maxnode=maxedge=0;total=0
 for step,s in enumerate(data["steps"]):
  rods=[r for r in s["rows"] if r["flags"]&8]
  assert len(rods)==2
  coords=[]
  for r in rods:
   a,b,e,marker,rod,collider,weight,radius,*_=r["scratch"]
   assert all(math.isfinite(v) for v in (a,b,e,marker,rod,collider,weight,radius))
   assert all(int(v)==v for v in (a,b,e,rod,collider)) and 0<=a<n and 0<=b<n and 0<=e<n-1 and rod==0
   assert marker>0 and 0<=weight<=1 and radius>0
   coords.append((int(a),int(b),int(e)))
  nodes=[x for a,b,e in coords for x in (a,b)];edges=[e for a,b,e in coords]
  node_span=max(nodes)-min(nodes);edge_span=max(edges)-min(edges)
  assert node_span<=2 and edge_span<=1
  maxnode=max(maxnode,node_span);maxedge=max(maxedge,edge_span);total+=1
  key=str(coords);patterns[key]=patterns.get(key,0)+1
 out[str(n)]={"steps":total,"node_span_max":maxnode,"edge_span_max":maxedge,"patterns":patterns,"trace_equal":True}
(p/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
