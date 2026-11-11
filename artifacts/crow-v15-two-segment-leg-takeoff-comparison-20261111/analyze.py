from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-two-segment-leg-takeoff-comparison-20261111');N=8;T=450
def read(arm):
 state={};sub=collections.defaultdict(lambda:[0.]*5);cover=collections.Counter();errors=[]
 with gzip.open(A/(arm+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);cover[e,t,k]+=1
    for i in range(5):sub[e,t][i]+=float(a[6+i])
   elif 'failed' in l.lower():errors.append(l.strip())
 try:
  out=json.loads((A/(arm+'.stdout')).read_text());failed=out['failed_environment_steps'];terminated=out['termination_count']
 except Exception as exc:failed=terminated=None;errors.append(str(exc))
 return state,sub,cover,errors,failed,terminated
p,pi,pc,pe,pf,pt=read('push');c,ci,cc,ce,cf,ct=read('control');rows=[]
def details(s,im,cov,e):
 complete=all((e,t) in s for t in range(1,T+1));missing=sum(cov[e,t,k]!=1 for t in range(1,T+1) for k in range(16));base=s[e,315]['q'][2] if (e,315) in s else None
 if not complete:return {'complete':False,'missing_substeps':missing,'event':None}
 ev=None
 for t in range(347,T+1):
  rise=s[e,t]['q'][2]-base;up=s[e,t]['v'][2];clear=all(im[e,u][2]+im[e,u][3]<=1e-7 for u in range(t-2,t+1));prior=max((im[e,u][0]+im[e,u][1] for u in range(316,t-1)),default=0.);nonfoot=sum(im[e,u][4]>1e-7 for u in range(316,t+1))
  if rise>=.03 and up>=.1 and clear and prior>=.5*.45004*9.81*.02 and nonfoot==0:
   ev={'step':t,'rise_m':rise,'vertical_speed_m_s':up,'maximum_prior_interval_upward_impulse_Ns':prior};break
 hits=[{'step':t,'indices':[i for i,v in enumerate(s[e,t]['v'][22:38]) if abs(v)>=11.999]} for t in range(1,T+1) if any(abs(v)>=11.999 for v in s[e,t]['v'][22:38])]
 return {'complete':complete,'missing_substeps':missing,'reference_root_height_m':base,'maximum_rise_m':max(s[e,t]['q'][2]-base for t in range(316,T+1)),'maximum_upward_speed_m_s':max(s[e,t]['v'][2] for t in range(316,T+1)),'unsupported_intervals':sum(im[e,t][2]+im[e,t][3]<=1e-7 for t in range(316,T+1)),'nonfoot_impulse_intervals':sum(im[e,t][4]>1e-7 for t in range(316,T+1)),'toe_speed_cap_hits':hits,'event':ev}
for e in range(N):
 diff=sum(p.get((e,t))!=c.get((e,t)) for t in range(1,301));a=details(p,pi,pc,e);b=details(c,ci,cc,e);rows.append({'lane':e,'pre_treatment_different_samples':diff,'push':a,'control':b,'pass':bool(diff==0 and a['complete'] and b['complete'] and a['missing_substeps']==0 and b['missing_substeps']==0 and a['event'] is not None and b['event'] is None and not a.get('toe_speed_cap_hits') and pf==cf==pt==ct==0 and not pe and not ce)})
report={'lanes':rows,'push_failed_environment_steps':pf,'control_failed_environment_steps':cf,'push_terminations':pt,'control_terminations':ct,'push_errors':pe[-5:],'control_errors':ce[-5:],'scope':'paired wing-suppressed leg launch development comparison; not flight handoff or landing'};(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
