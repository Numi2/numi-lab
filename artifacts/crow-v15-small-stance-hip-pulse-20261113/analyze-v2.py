from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-small-stance-hip-pulse-20261113');state={};balance={};event={};target={};pulse={};count=collections.Counter();errors=[]
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0] in ('step','balance','event','rightHipCommand','pulse'):
   e,t=int(x[1]),int(x[2]);k=e,t
   if x[0]=='step':
    q=x.index('q');v=x.index('v');state[k]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x[0]=='balance':balance[k]=list(map(float,x[3:]))
   elif x[0]=='event':event[k]=list(map(float,x[3:]))
   elif x[0]=='rightHipCommand':target[k]=float(x[3])
   else:pulse[k]=list(map(int,x[3:]))
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0]=='CROW_TWO_TOE_SUBSTEP':
   t,sub,e=map(int,x[1:4]);count[e,t,sub]+=1
   if int(x[4])>int(x[5]):errors.append('contact overflow '+str((e,t,sub)))
  elif x[0]=='CROW_SUBSTEP_ERROR' or x[0]=='failure' or 'OBSERVER_ERROR' in x[0]:errors.append(line.strip())
onsets=[next((t for t in range(235) if pulse.get((e,t),[0])[0]),None) for e in (0,1)];on=onsets[0] if onsets[0] is not None and onsets[0]==onsets[1] else None
pre=[max(abs(a-b) for a,b in zip(state[0,t],state[1,t])) for t in range(on or 0) if (0,t) in state and (1,t) in state];missing=sum(count[e,t,k]!=1 for e in (0,1) for t in range(1,236) for k in range(32));rows=[]
if on is not None:
 for t in range(on,min(on+11,235)):
  if not all((e,t) in balance and (e,t) in event and (e,t) in target and (e,t) in pulse for e in (0,1)):continue
  b0,b1=balance[0,t],balance[1,t];a0,a1=event[0,t],event[1,t]
  def support(a):return a[3]>.001 and a[6]>.000001 and a[7]<-.000001
  rows.append({'zero_based_step':t,'relative_x_difference_m':b0[2]-b1[2],'relative_vx_difference_m_s':b0[4]-b1[4],'right_hip_target_difference_rad':target[0,t]-target[1,t],'right_opposing_support':[support(a0),support(a1)],'pulse_active':[pulse[0,t][0],pulse[1,t][0]]})
pulse_counts=[sum(bool(pulse.get((e,t),[0])[0]) for t in range(235)) for e in (0,1)]
command_ok=bool(rows and abs(rows[0]['right_hip_target_difference_rad']-.006)<1e-5 and rows[0]['pulse_active']==[1,1] and pulse_counts==[1,1])
out={'pulse_counts':pulse_counts,'matched_command_check':command_ok,'pulse_onsets_zero_based':onsets,'prefix_state_pairs':len(pre),'prefix_max_abs_qv_difference':max(pre,default=None),'missing_substeps':missing,'errors':errors,'rows':rows,'usable_for_sign_identification':bool(on is not None and len(pre)==on and max(pre,default=1)==0 and command_ok and missing==0 and not errors and len(rows)==11 and all(all(x['right_opposing_support']) for x in rows)),'scope':'matched target pulses on a constructed stance; balance values precede each step and support values are accepted endpoint observations'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
