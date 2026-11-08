import hashlib,json,math,pathlib
import numpy as np
p=pathlib.Path(__file__).resolve().parent
manifest=json.loads((p/'runs.json').read_text());runs=manifest['runs']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert manifest['probe_sha256']==sha(p/'probe')
assert len(runs)==14 and all(r['exit']==0 for r in runs)
for r in runs:
 assert r['output_sha256']==sha(p/(r['name']+'.json'))
 assert r['trace_sha256']==sha(p/(r['name']+'.stderr'))
 if r['name']=='baseline':
  assert r['metallib_sha256']==sha(pathlib.Path('/Users/numi/numi-lab-ccd-thin-evidence-20261107/cold-four-sliding-v1/candidate.metallib'))
 else:
  slot=int(r['name'].split('-')[1]);assert r['metallib_sha256']==sha(p/f'instrumented-{slot}.metallib')

def read(name):
 data=json.loads((p/(name+'.json')).read_text())['runs'];events=[];case=None
 for line in (p/(name+'.stderr')).read_text().splitlines():
  if line.startswith('ccd_case='):case=json.loads(line.split('=',1)[1])
  elif line.startswith('ccd_event_trace='):
   assert case is not None
   events.append((case,json.loads(line.split('=',1)[1])))
 return data,events
baseline,baseevents=read('baseline');assert len(baseline)==2 and len(baseevents)==16
captures={1:{},.5:{}}
for slot in range(13):
 values,events=read(f'slot-{slot}');assert len(values)==2 and len(events)==len(baseevents)
 for a,b in zip(baseline,values):
  a=dict(a);b=dict(b);a.pop('elapsed_with_initialization',None);b.pop('elapsed_with_initialization',None);assert a==b,('result mismatch',slot,a,b)
 for (ca,a),(cb,b) in zip(baseevents,events):
  assert ca==cb
  a=dict(a);b=dict(b)
  if b['pair_flags']&128:
   if b['pass']==1:
    scale=ca['scale'];raw=b['pair_time'];previous=captures[scale].get(slot,raw);assert previous==raw;captures[scale][slot]=raw
   a.pop('pair_time');b.pop('pair_time')
  assert a==b,('trace mismatch',slot,ca,a['pass'])

def norm(v):return float(np.linalg.norm(v))
def measure(params,lo,hi,allcorners):
 q0=np.array(params['q0'],dtype=float);q1=np.array(params['q1'],dtype=float);d=q1-q0
 dd=float(d@d);amin=max(lo,min(hi,-float(q0@d)/dd)) if dd else lo
 dmin=min(float((q0+amin*d)@(q0+amin*d)),float((q0+lo*d)@(q0+lo*d)),float((q0+hi*d)@(q0+hi*d)))
 assert dmin>0
 n0=norm(q0);c=float(q0@q1)/n0;sn=norm(q1-q0*(c/n0));angle=math.atan2(sn,c)
 grid=np.linspace(lo,hi,100001);q=q0+grid[:,None]*d;q/=np.linalg.norm(q,axis=1)[:,None]
 if angle>1e-6:
  time=np.arctan2(grid*sn,(1-grid)*n0+grid*c)/angle
  tmax=n0*sn/(angle*dmin)
 else:
  den=(1-grid)*n0+grid*c;time=grid*c/den
  denmin=min((1-lo)*n0+lo*c,(1-hi)*n0+hi*c);tmax=c*n0/denmin**2
 x,y,z,w=q.T;row=np.stack((2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),axis=1)
 cy=params['y0']+time*(params['y1']-params['y0'])-params['plane_y'];ext=np.array(params['extents'])
 if allcorners:gap=cy-np.abs(row)@ext
 else:gap=cy-row@ext
 index=int(np.argmin(gap));minimum=float(gap[index])
 lip=abs(params['y1']-params['y0'])*tmax+norm(ext)*2*norm(d)/math.sqrt(dmin)
 lower=minimum-lip*(hi-lo)/200000.-2e-7
 return {'sample_min_gap_m':minimum,'sample_argmin_alpha':float(grid[index]),'sampling_lower_with_roundoff_guard_m':lower,'sample_count':100001,'lipschitz_per_alpha':lip}
result={'schema':'numi.ccd.pending-independent-geometry.v1','comparison_pass':True,'trace_count_per_selector':len(baseevents),'scales':[],'limitations':['Projected candidate sweep only; rejected state is never accepted.','Whole-interval sampling lower bound includes a 0.2 micrometre implementation/decimal guard and is a diagnostic, not a verified production certificate.','The certificate halted in corner 0; other corners were not necessarily processed.']}
for r in baseline:
 scale=r['scale'];cap=captures[scale];assert len(cap)==13,(scale,len(cap))
 assert abs(cap[11][0]-1)<1e-9 and abs(cap[12][0])<1e-9 and abs(cap[12][1])<1e-9,'fixture plane is not fixed positive-y'
 par={'q0':cap[6]+cap[7],'q1':cap[8]+cap[9],'y0':cap[5][0],'y1':cap[5][1],'plane_y':cap[11][1],'guard_m':cap[10][0],'tolerance_m':cap[10][1],'extents':r['half_extents']}
 intervals=[cap[i] for i in range(5) if cap[i]!=[-1,-1]]
 assert len(intervals)==(1 if scale==1 else 5),intervals
 for lo,hi in intervals:assert 0<=lo<hi<=1
 entry={'scale':scale,'capture':par,'pending':[],'full_remainder_corner0':measure(par,0,1,False),'full_remainder_all_corners':measure(par,0,1,True)}
 for lo,hi in intervals:entry['pending'].append({'alpha':[lo,hi],'corner0':measure(par,lo,hi,False),'all_corners':measure(par,lo,hi,True)})
 result['scales'].append(entry)
(p/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
