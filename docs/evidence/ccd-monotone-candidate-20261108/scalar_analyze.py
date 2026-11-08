import hashlib,json,math,pathlib
import numpy as np
p=pathlib.Path(__file__).parent;x=json.loads((p/'scalar-observations.json').read_text());assert x['seed']==0x20261108 and len(x['cases'])==2048
out=[];crossing_controls=0;fallback=0
for c in x['cases']:
 q0=np.asarray(c['q0']);q1=np.asarray(c['q1']);q1= -q1 if q0@q1<0 else q1;d=q1-q0;dd=float(d@d);amin=max(0.,min(1.,-float(q0@d)/dd)) if dd else 0.;dmin=float((q0+amin*d)@(q0+amin*d));assert dmin>0
 n0=float(np.linalg.norm(q0));cc=float(q0@q1)/n0;sn=float(np.linalg.norm(q1-q0*(cc/n0)));angle=math.atan2(sn,cc)
 fast=c['handled']==1 and c['clear']==1 and c['work']==0
 n=65537 if fast else 2049;alpha=np.linspace(0,1,n);q=q0+alpha[:,None]*d;q/=np.linalg.norm(q,axis=1)[:,None]
 if angle>1e-6:t=np.arctan2(alpha*sn,(1-alpha)*n0+alpha*cc)/angle;vmax=n0*sn/(angle*dmin)
 else:
  den=(1-alpha)*n0+alpha*cc;t=alpha*cc/den;vmax=cc*n0/min(n0,cc)**2
 a,b,z,w=q.T;row=np.stack((2*(a*b+z*w),1-2*(a*a+z*z),2*(b*z-a*w)),axis=1);half=np.asarray(c['half']);gap=c['y0']+t*(c['y1']-c['y0'])-c['offset']-np.abs(row)@half;idx=int(np.argmin(gap));minimum=float(gap[idx])
 if not fast:
  fallback+=1
  if minimum < -2e-7*c['scale']:crossing_controls+=1
  continue
 lipschitz=abs(c['y1']-c['y0'])*vmax+2*np.linalg.norm(half)*np.linalg.norm(d)/math.sqrt(dmin)
 guard=max(2e-7*c['scale'],64*np.finfo(float).eps*(abs(c['offset'])+abs(c['y0'])+abs(c['y1'])+float(np.linalg.norm(half))))
 lower=minimum-lipschitz/(2*(n-1))-guard
 out.append({'id':c['id'],'scale':c['scale'],'offset':c['offset'],'sample_min_gap_m':minimum,'sample_argmin_alpha':float(alpha[idx]),'lower_bound_m':lower,'required_tolerance_m':c['tolerance'],'margin_m':lower-c['tolerance'],'pass':bool(lower>c['tolerance'])})
result={'schema':'numi.ccd.monotone-native-screen.v1','cases':len(x['cases']),'fast_clears':len(out),'fallback_cases':fallback,'sampled_crossing_controls':crossing_controls,'all_fast_clears_pass_independent_bound':bool(out) and all(r['pass'] for r in out),'cases_detail':out,'limitations':['Deterministic finite sample and diagnostic Lipschitz enclosure, not a proof for all geometry.','Only static Y-plane and centered single-box projected sweeps were tested.','A 0.2 micrometre scaled numerical allowance is not a formal enclosure of all Metal rounding.','No full-step contact response or transaction claims.']}
(p/'scalar-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases_detail'},indent=2));print('smallest_margins',sorted(out,key=lambda z:z['margin_m'])[:5])
