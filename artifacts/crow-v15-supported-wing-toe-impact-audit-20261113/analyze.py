from pathlib import Path
import gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-supported-wing-toe-impact-audit-20261113';A.mkdir(exist_ok=True);base=R/'crow-v15-supported-wing-transition-20261113';p=base/'push-wing-20261126-rebound-supportedwing/run.stderr.gz';events={e:[] for e in range(2,8)};starts={};pre={};foot={};pairs={};q={}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if not z:continue
  typ=z[0]
  if typ not in ('CROW_TOE_V_SUBSTEP','CROW_KIN_STATE','CROW_PRESOLVE_V','CROW_TWO_TOE_SUBSTEP','CROW_GROUND_CONTACT'):continue
  t,k,e=map(int,z[1:4]);key=e,t,k
  if e not in events:continue
  if typ=='CROW_TOE_V_SUBSTEP' and 348<=t<=400:
   v=list(map(float,z[4:]));assert len(v)==16
   for j,x in enumerate(v):
    if abs(x)>=11.99:events[e].append((t,k,22+j,x))
  elif typ=='CROW_KIN_STATE':
   nq,nv=map(int,z[4:6]);v=list(map(float,z[6:]));assert len(v)==nq+nv;starts[key]=v[nq:];q[key]=v[:nq]
  elif typ=='CROW_PRESOLVE_V':pre[key]=list(map(float,z[5:]))
  elif typ=='CROW_TWO_TOE_SUBSTEP':foot[key]=list(map(float,z[6:11]))
  elif typ=='CROW_GROUND_CONTACT':pairs.setdefault(key,set()).add((int(z[4]),int(z[5])))
analysis=json.loads((base/'analysis.json').read_text());rows=[]
for e in range(2,8):
 assert events[e];t,k,j,v=min(events[e],key=lambda x:(x[0],x[1],x[2]));key=e,t,k;release=analysis['rows'][e]['push']['first_three_unloaded'];row={'lane':e,'first_threshold_step':t,'substep':k,'velocity_dof':j,'end_velocity_rad_s':v,'first_three_unloaded_step':release,'foot_impulses':foot.get(key),'ground_body_pairs':sorted(pairs.get(key,[]))}
 if key in pre and key in starts:row.update({'start_velocity_rad_s':starts[key][j],'precontact_velocity_rad_s':pre[key][j],'post_presolve_delta_rad_s':v-pre[key][j],'q_coordinate_rad':q[key][j+1]})
 rows.append(row)
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc first toe-speed threshold in supported-wing push arm, used seed and correlated lanes. Precontact state and body pairs exist only for substeps 348–375. Recorded post-presolve change can include contact resolution and later substep processing; no unique cause is established.','rows':rows},indent=2)+'\n')
for x in rows:print(x)
