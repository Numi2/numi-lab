from pathlib import Path
import gzip,json,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');B=R/'crow-v15-supported-wing-transition-20261113';A=R/'crow-v15-supported-toe-dynamics-audit-20261113';A.mkdir(exist_ok=True)
first=json.loads((R/'crow-v15-supported-wing-toe-impact-audit-20261113/analysis.json').read_text())['rows']; events={x['lane']:x for x in first if x['lane'] in (2,3,4,7)}; out={}; summaries=[]
for arm in ('push','hold'):
 p=B/(arm+'-wing-20261126-rebound-supportedwing/run.stderr.gz');rows={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if not z or z[0] not in ('CROW_TOE_V_SUBSTEP','CROW_KIN_STATE','CROW_PRESOLVE_V','CROW_GROUND_CONTACT'):continue
   t,k,e=map(int,z[1:4]);
   if e not in events or not (351<=t<=355) or (t==355 and k>events[e]['substep']):continue
   key=(e,t,k);row=rows.setdefault(key,{'lane':e,'tick':t,'substep':k});j=events[e]['velocity_dof']
   if z[0]=='CROW_TOE_V_SUBSTEP':row['end']=float(z[4+j-22])
   elif z[0]=='CROW_PRESOLVE_V':row['pre']=float(z[5+j])
   elif z[0]=='CROW_KIN_STATE':
    nq,nv=map(int,z[4:6]);row['q']=float(z[6+j+1]);row['start']=float(z[6+nq+j])
   else:
    a,b=map(int,z[4:6]);imp=float(z[20]);
    if imp>1e-7:row['contacts']=row.get('contacts',[])+[[a,b,imp]]
 out[arm]=[v for k,v in sorted(rows.items()) if 'end' in v and 'pre' in v and 'start' in v]
 for e,ev in events.items():
  a=[x for x in out[arm] if x['lane']==e];assert len(a)==4*16+ev['substep']+1,(arm,e,len(a)); pre=sum(x['pre']-x['start'] for x in a);post=sum(x['end']-x['pre'] for x in a);bridge=max(abs(a[i]['start']-a[i-1]['end']) for i in range(1,len(a)));start=a[0]['start'];end=a[-1]['end'];onset=next(({'tick':x['tick'],'substep':x['substep'],'velocity':x['end']} for x in a if x['end']<=-3),None)
  summaries.append({'arm':arm,'lane':e,'dof':ev['velocity_dof'],'substeps':len(a),'start_velocity':start,'end_velocity':end,'sum_precontact_velocity_change':pre,'sum_post_presolve_velocity_change':post,'sum_residual':end-start-pre-post,'maximum_inter_substep_velocity_discontinuity':bridge,'start_q':a[0]['q'],'end_q':a[-1]['q'],'first_negative_3_rad_s':onset,'substeps_with_positive_logged_ground_impulses':sum(bool(x.get('contacts')) for x in a)})
(A/'analysis.json').write_text(json.dumps({'scope':'Same used seed, correlated lanes, selected front proximal joints; accepted start, predicted presolve, postsolve and positive ground pair impulses, 351.0 through each first push threshold. Post-presolve contains contact and subsequent processing, not an isolated force. Velocities are generalized and cannot attribute force to one contact.','source_sha256':{arm:hashlib.sha256((B/(arm+'-wing-20261126-rebound-supportedwing/run.stderr.gz')).read_bytes()).hexdigest() for arm in ('push','hold')},'summaries':summaries,'substeps':out},indent=2)+'\n')
for x in summaries:print(x)
