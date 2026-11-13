from pathlib import Path
import json,math
A=Path(__file__).resolve().parent;expect=json.loads((A/'expected-poses.json').read_text());base={};cand={0:[],1:[]}
for l in (A/'results.tsv').read_text().splitlines():
 x=l.split();e,t=int(x[1]),int(x[2]);assert t==expect['selected_steps_zero_based'][str(e)]
 if x[0]=='B':
  k=e,x[3];assert k not in base;base[k]=list(map(float,x[4:]));assert len(base[k])==7
 elif x[0]=='C':
  v=list(map(float,x[3:]));assert len(v)==10 and all(map(math.isfinite,v));cand[e].append(v)
 else:raise AssertionError(x[0])
assert len(base)==18 and all(len(cand[e])==125 for e in (0,1));errors=[]
for (e,n),v in base.items():errors.extend(abs(a-b) for a,b in zip(v,expect['poses'][str(e)][n]))
assert len(errors)==126 and max(errors)<=1e-6
out={'base_pose_components':len(errors),'max_abs_pose_error':max(errors),'lanes':{}}
for e,rows in cand.items():
 original=next(x for x in rows if x[:3]==[0,0,0]);best=max(rows,key=lambda x:x[3]);valid=[x for x in rows if x[3]>=.001];conservative=[x for x in valid if max(map(abs,x[:3]))<=.15];out['lanes'][str(e)]={'candidate_count':len(rows),'initial_gap_m':original[3],'max_gap_m':best[3],'best_offsets_hip_knee_ankle_rad':best[:3],'candidates_at_least_1mm':len(valid),'within_0_15rad_and_1mm':len(conservative),'smallest_norm_clearance_candidate':min(conservative or valid,key=lambda x:sum(v*v for v in x[:3]))[:4] if valid else None}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
