from pathlib import Path
import gzip,json,math,numpy as np
A=Path('artifacts/crow-v15-stance-toe-workspace-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114/trace.log.gz')
meta={r['id']:r for r in json.loads((A/'inputs.json').read_text())['records']};bodies={};original={}
for line in (A/'output.tsv').open():
 z=line.split()
 if z and z[0]=='body' and z[3].startswith('crow_right_') and ('toe' in z[3] or 'foot' in z[3]):bodies.setdefault(int(z[2]),{})[z[3]]=list(map(float,z[4:]))
for line in gzip.open(P,'rt'):
 z=line.split()
 if z and z[0]=='pose' and z[1]=='0' and z[2]=='304' and z[3].startswith('crow_right_') and ('toe' in z[3] or 'foot' in z[3]):original[z[3]]=list(map(float,z[4:]))
def gap(obj):
 x,y,z,w=obj[3:];P=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);Q=np.eye(3);d=np.array([0.,0.,-.055])-np.array(obj[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];return max(abs(d@(u/np.linalg.norm(u)))-sum(abs(P.T@(u/np.linalg.norm(u)))*[.009,.0025,.0092])-sum(abs(Q.T@(u/np.linalg.norm(u)))*[.025,.72,.055]) for u in axes if np.linalg.norm(u)>1e-8)
errs={name:math.dist(bodies[0][name][:3],value[:3]) for name,value in original.items() if name in bodies[0]};missing=sorted(set(original)-set(bodies.get(0,{})));rows=[]
for n,info in meta.items():
 if info['kind']=='original':continue
 names=[k for k in bodies.get(n,{}) if k.startswith('crow_right_'+('front_toe' if info['kind']=='front' else 'rear_toe'))]
 gaps=[gap(bodies[n][k]) for k in names];expected=6 if info['kind']=='front' else 2
 if len(gaps)!=expected:raise RuntimeError('missing segment')
 rows.append({'id':n,'kind':info['kind'],'angles':info['angles'],'min_gap_m':min(gaps),'max_gap_m':max(gaps),'max_penetration_m':max(0.,-min(gaps)),'near_surface_0_5mm_with_under_1mm_perch_penetration':bool(any(abs(v)<=.0005 for v in gaps) and min(gaps)>=-.001)})
summ={}
for k in ('front','rear'):
 r=[x for x in rows if x['kind']==k];good=[x for x in r if x['near_surface_0_5mm_with_under_1mm_perch_penetration']];summ[k]={'grid_configurations':len(r),'near_surface_configurations':len(good),'examples':good[:8]}
out={'mapping':{'recorded_names':len(original),'max_position_error_m':max(errs.values(),default=None),'missing_names':missing},'summary':summ,'rows':rows,'scope':'fixed-posture oriented-box kinematic screen; no dynamics, self-collision or path feasibility'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
