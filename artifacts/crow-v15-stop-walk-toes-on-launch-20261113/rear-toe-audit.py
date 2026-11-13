from pathlib import Path
import collections,gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-stop-walk-toes-on-launch-20261113';p=A/'push-nowings-20261127-rebound-no-walk-toes/run.stderr.gz';kin={};pre={};post={};contacts={}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if not z:continue
  if z[0]=='CROW_KIN_STATE':
   t,k,e,nq,nv=map(int,z[1:6]);v=list(map(float,z[6:]));assert nq==41 and nv==40 and len(v)==81;kin[e,t,k]=v
  elif z[0]=='CROW_PRESOLVE_V':
   t,k,e,n=map(int,z[1:5]);v=list(map(float,z[5:]));assert n==40 and len(v)==40;pre[e,t,k]=v
  elif z[0]=='CROW_TOE_V_SUBSTEP':
   t,k,e=map(int,z[1:4]);post[e,t,k]=list(map(float,z[4:]))
  elif z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4]);contacts[e,t,k]=list(map(float,z[11:27]))
rows=[]
for (e,t,k),v in post.items():
 if e<2 or not 348<=t<=375:continue
 for j in (6,14):
  if abs(v[j])<11.99:continue
  key=e,t,k;assert key in kin and key in pre and key in contacts
  rows.append({'lane':e,'step':t,'substep':k,'toe_index':j,'begin_velocity_rad_s':kin[key][41+22+j],'pre_contact_velocity_rad_s':pre[key][22+j],'post_velocity_rad_s':v[j],'same_body_normal_impulse_ns':contacts[key][j]})
assert rows;out={'event_count':len(rows),'events':rows,'pre_contact_at_or_above_threshold':sum(abs(x['pre_contact_velocity_rad_s'])>=11.99 for x in rows),'same_body_positive_contact_impulse':sum(x['same_body_normal_impulse_ns']>1e-7 for x in rows),'scope':'Posthoc selected events with post-substep rear proximal toe speeds >=11.99 rad/s; beginning and candidate pre-contact buffers are logged separately. Same-body normal impulse is only one contact proxy, may omit other constraints. Threshold proximity does not prove active clipping; comparisons are not independent.'};(A/'rear-toe-audit.json').write_text(json.dumps(out,indent=2)+'\n');print('events',len(rows),'pre-contact near limit',out['pre_contact_at_or_above_threshold'],'same-body normal impulse',out['same_body_positive_contact_impulse']);print('pre velocity range',min(x['pre_contact_velocity_rad_s'] for x in rows),max(x['pre_contact_velocity_rad_s'] for x in rows));print('sample',rows[:3])
