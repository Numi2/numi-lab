from pathlib import Path
import gzip,json,numpy as np
A=Path('artifacts/crow-v15-perch-placement-workspace-20261114');P=Path('artifacts/crow-v15-stance-toe-workspace-20261114');meta={r['id']:r for r in json.loads((A/'inputs.json').read_text())['records']};old={r['id']:r for r in json.loads((P/'analysis.json').read_text())['rows']};bodies={}
for line in gzip.open(A/'output.tsv.gz','rt'):
 z=line.split()
 if z and z[0]=='body' and z[3].startswith('crow_right_') and 'toe' in z[3]:bodies.setdefault(int(z[2]),{})[z[3]]=list(map(float,z[4:]))
def gap(o):
 x,y,z,w=o[3:];p=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(o[:3]);axes=[*p.T,*q.T]+[np.cross(p[:,i],q[:,j]) for i in range(3) for j in range(3)];return float(max(abs(d@(u/np.linalg.norm(u)))-sum(abs(p.T@(u/np.linalg.norm(u)))*[.009,.0025,.0092])-sum(abs(q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8))
counts={};parity=[];missing=[];examples={}
for n,r in meta.items():
 names=[k for k in bodies.get(n,{}) if k.startswith('crow_right_'+('front_toe' if r['kind']=='front' else 'rear_toe'))]
 if len(names)!=(6 if r['kind']=='front' else 2):missing.append(n);continue
 gaps=[gap(bodies[n][k]) for k in names];isgood=any(abs(x)<=.0005 for x in gaps) and min(gaps)>=-.001;key=str(r['offset_m'])+':'+r['kind'];counts.setdefault(key,{'total':0,'near_surface':0});counts[key]['total']+=1;counts[key]['near_surface']+=int(isgood)
 if isgood and len(examples.setdefault(key,[]))<3:examples[key].append({'angles':r['angles'],'min_gap_m':min(gaps),'max_gap_m':max(gaps)})
 if r['offset_m']==0.:parity.append(abs(min(gaps)-old[r['prior_id']]['min_gap_m']))
out={'counts':counts,'examples':examples,'zero_offset_parity_count':len(parity),'zero_offset_max_min_gap_difference_m':max(parity,default=None),'missing':missing,'scope':'Rigid whole-model translation, fixed joint pose, SAT geometry; no dynamics, contact impulse, self-collision, physical trajectory or grip.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
