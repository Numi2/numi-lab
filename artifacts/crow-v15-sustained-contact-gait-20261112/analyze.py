from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-sustained-contact-gait-20261112');arms={}
for variant in ('control','hip'):
 D=A/('candidate-20261126-rebound-'+variant);meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];state={};sub={};toe=collections.Counter();toe_seen=set();overflow=[];errors=[]
 with gzip.open(D/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if t<=meta['steps_by_environment'][e] and t<=500:state[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=meta['steps_by_environment'][e] and t<=500:
     assert (e,t,k) not in sub;sub[e,t,k]=z[4:]
     if int(z[4])>int(z[5]):overflow.append([e,t,k])
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=meta['steps_by_environment'][e] and 201<=t<=500:
     assert (e,t,k) not in toe_seen;toe_seen.add((e,t,k))
     if max(abs(float(v)) for v in z[4:])>=11.99:toe[e]+=1
   elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
 rows=[]
 for e in range(8):
  end=min(500,meta['steps_by_environment'][e]);coverage=sum((e,t,k) in sub for t in range(201,end+1) for k in range(16));seq=[sub[e,t,k] for t in range(201,end+1) for k in range(16) if (e,t,k) in sub]
  unloaded=sum(float(x[4])+float(x[5])<=1e-7 for x in seq);nonfoot=sum(float(x[6])>1e-7 for x in seq)
  bouts={}
  for side,idx in (('left',4),('right',5)):
   loaded=[i for i,x in enumerate(seq) if float(x[idx])>1e-7];groups=[]
   for i in loaded:
    if not groups or i-groups[-1][-1]>3:groups.append([i])
    else:groups[-1].append(i)
   bouts[side]=sum(len(g)>=32 and g[0]>0 and g[-1]<len(seq)-1 for g in groups)
  dx=state[e,end]['q'][0]-state[e,200]['q'][0] if (e,end) in state and (e,200) in state else None
  rows.append({'lane':e,'end':end,'failure':meta['physical_failure_by_environment'][e],'contact_substeps':coverage,'expected_substeps':max(0,end-200)*16,'both_unloaded_substeps':unloaded,'nonfoot_impulse_substeps':nonfoot,'toe_observed_substeps':sum((e,t,k) in toe_seen for t in range(201,end+1) for k in range(16)),'toe_cap_substeps':toe[e],'support_bouts_at_least_40ms':bouts,'dx_m':dx})
 arms[variant]={'rows':rows,'state':state,'sub':sub,'errors':errors,'overflow':overflow}
rows=[]
for e in range(8):
 x,y=arms['control'],arms['hip'];a,b=x['rows'][e],y['rows'][e];states=sum(x['state'].get((e,t))==y['state'].get((e,t)) and (e,t) in x['state'] and (e,t) in y['state'] for t in range(1,201));contacts=sum(x['sub'].get((e,t,k))==y['sub'].get((e,t,k)) and (e,t,k) in x['sub'] and (e,t,k) in y['sub'] for t in range(1,201) for k in range(16));qualified=states==200 and contacts==3200 and b['end']==500 and b['contact_substeps']==4800 and not b['failure'] and not b['nonfoot_impulse_substeps'] and b['toe_observed_substeps']==4800 and not b['toe_cap_substeps'] and not x['errors'] and not y['errors'] and not x['overflow'] and not y['overflow'] and b['both_unloaded_substeps']<=a['both_unloaded_substeps'] and b['dx_m'] is not None and b['dx_m']>=.15
 rows.append({'lane':e,'prewalk_matching_endpoints':states,'prewalk_matching_contact_substeps':contacts,'control':a,'hip':b,'engineering_screen':qualified})
r={'rows':rows,'errors':{v:arms[v]['errors'] for v in arms},'overflow':{v:arms[v]['overflow'] for v in arms},'scope':'Isolated 4–10 second walking diagnostic, two used-seed arms. Aggregate normal impulse is thresholded 1e-7 N s; contact gaps and foot speed metrics do not prove tracked material slip. Toe speed cap uses 11.99 rad/s, body/wing caps not observed. No navigation, takeoff or biological qualification.'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
for x in rows:print(x)
print('errors',r['errors'],'overflow',r['overflow'])
