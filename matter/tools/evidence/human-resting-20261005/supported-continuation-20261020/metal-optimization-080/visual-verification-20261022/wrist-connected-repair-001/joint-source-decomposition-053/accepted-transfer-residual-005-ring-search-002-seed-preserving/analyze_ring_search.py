#!/usr/bin/env python3
import csv, json, math, hashlib, statistics
from pathlib import Path
base=Path(__file__).resolve().parent
root=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
names={int(k):v for k,v in json.loads((root/"joint-source-decomposition-053/accepted-transfer-residual-001/owner-names.json").read_text()).items()}
def region(name):
 n=name.lower()
 if "toes" in n: return "foot_toes"
 if any(k in n for k in ("humerus","scapula","clavicle")): return "shoulder"
 if any(k in n for k in ("radius","ulna")): return "elbow_forearm"
 if any(k in n for k in ("capitate","scaphoid","lunate","triquetrum","trapezium","trapezoid","hamate","firstmc","secondmc","thirdmc","fourthmc","fifthmc","proxph","midph","distph","toes")): return "wrist_hand"
 return None
def qtile(v,p):
 if not v:return 0.0
 w=sorted(v);x=(len(w)-1)*p;lo=math.floor(x);hi=math.ceil(x)
 return w[lo]*(hi-x)+w[hi]*(x-lo)
def load_ring(r):
 d=base/"results"/f"ring-{r}"
 with (d/"selected_transfer_map.csv").open() as f:
  mapping={int(x["vertex"]):x for x in csv.DictReader(f)}
 bycase={}
 with (d/"source_lbs_minus_transfer_baseline.csv").open() as f:
  for x in csv.DictReader(f):
   i=int(x["vertex"]);name=names.get(int(x["donor"]),str(x["donor"]));reg=region(name)
   if reg is None:continue
   x["vertex"]=i;x["donor_name"]=name;x["region"]=reg;x["norm"]=float(x["residual_norm_m"])
   bycase.setdefault(x["case"],{})[i]=x
 return mapping,bycase
rings={r:load_ring(r) for r in (1,2,3)}
cases=["reference","shoulder-elevation-050","elbow-flexion-080"]
out={"scope":"source-registration static AcceptedFEMSkinTransfer mapping; not FEM dynamics or tissue qualification","ring_semantics":"1=legacy incident faces; 2=incident set plus one shared-edge face step; 3=plus two shared-edge steps. Original incident seeds retained. Expanded faces require detail donor corner and same edge-connected surface component; stable source-face ID then row tie-break.","by_region":{},"map_changes_from_ring1":{},"changes_vs_ring1":{}}
for r in (1,2,3):
 mapping,bycase=rings[r]
 if r>1:
  m1=rings[1][0]
  changed=[i for i in mapping if mapping[i]["coarse_face"]!=m1[i]["coarse_face"] or any(mapping[i][f"coarse{k}"]!=m1[i][f"coarse{k}"] for k in range(3)) or any(abs(float(mapping[i][f"w{k}"])-float(m1[i][f"w{k}"]))>0 for k in range(3))]
  out["map_changes_from_ring1"][str(r)]={"rows_changed":len(changed),"active_rows_changed":sum(mapping[i]["used"]=="1" for i in changed),"direct_source_rows_changed":sum(mapping[i]["used"]=="1" and float(mapping[i]["projection_distance_m"])==0 for i in changed)}
 for reg in ("shoulder","elbow_forearm","wrist_hand","foot_toes"):
  regout={}
  for case in cases:
   rows=[x for x in bycase[case].values() if x["region"]==reg]
   norms=[x["norm"] for x in rows]
   ids=[x["vertex"] for x in rows]
   dists=[float(mapping[i]["projection_distance_m"]) for i in ids]
   worst=max(rows,key=lambda x:x["norm"]) if rows else None
   regout[case]={"count":len(rows),"residual_mm":{"median":1000*qtile(norms,.5),"p95":1000*qtile(norms,.95),"p99":1000*qtile(norms,.99),"max":1000*max(norms,default=0)},"projection_mm":{"p95":1000*qtile(dists,.95),"max":1000*max(dists,default=0),">5mm":sum(x>.005 for x in dists),">10mm":sum(x>.01 for x in dists)},"worst":None if not worst else {"vertex":worst["vertex"],"donor":worst["donor_name"],"residual_mm":1000*worst["norm"],"projection_mm":1000*float(mapping[worst["vertex"]]["projection_distance_m"]),"face":int(mapping[worst["vertex"]]["coarse_face"]),"stable_source_face":int(mapping[worst["vertex"]]["stable_source_face"])}}
  out["by_region"].setdefault(reg,{})[str(r)]=regout
for r in (2,3):
 for case in cases:
  for reg in ("shoulder","elbow_forearm","wrist_hand"):
   a=rings[1][1][case];b=rings[r][1][case]
   ids=set(a)&set(b);pairs=[(a[i]["norm"],b[i]["norm"]) for i in ids if a[i]["region"]==reg and b[i]["region"]==reg]
   deltas=[new-old for old,new in pairs]
   out["changes_vs_ring1"].setdefault(str(r),{}).setdefault(case,{})[reg]={"rows":len(pairs),"improved_any":sum(d<0 for d in deltas),"improved_gt_1mm":sum(d<-.001 for d in deltas),"worsened_gt_1mm":sum(d>.001 for d in deltas),"median_delta_mm":1000*qtile(deltas,.5),"p95_abs_delta_mm":1000*qtile([abs(d) for d in deltas],.95),"max_improvement_mm":-1000*min(deltas,default=0),"max_worsening_mm":1000*max(deltas,default=0)}
out["input_hashes"]={}
for r in (1,2,3):
 d=base/"results"/f"ring-{r}"
 for name in ("selected_transfer_map.csv","source_lbs_minus_transfer_baseline.csv"):
  p=d/name;out["input_hashes"][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
p=base/"regional-ring-analysis-002.json";p.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
