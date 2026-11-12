from pathlib import Path
import collections,gzip,json
A=Path("artifacts/crow-v15-active-contact-failure-observer-20261112");D=A/"candidate-20261126-rebound-hip";meta=json.loads((D/"run.stdout").read_text())["first_episode_navigation"];ends=meta['steps_by_environment'];status=collections.defaultdict(list);contacts=collections.defaultdict(list)
with gzip.open(D/"run.stderr.gz","rt") as f:
 for l in f:
  z=l.split()
  if not z:continue
  if z[0]=="CROW_CONTACT_STATUS":
   t,k,e,req,active,code=map(int,z[1:7])
   if t<=ends[e]:status[e].append({'step':t,'substep':k,'required':req,'active':active,'code':code})
  elif z[0]=="CROW_ACTIVE_CONTACT":
   t,k,e,a,b=map(int,z[1:6])
   if t<=ends[e]:contacts[e].append({'step':t,'substep':k,'bodyA':a,'bodyB':b,'normal_impulse':float(z[6]),'point_xyz':list(map(float,z[7:10])),'separation':float(z[10])})
rows=[]
for e in range(8):
 end=ends[e];last=[x for x in status[e] if end-2<=x['step']<=end];latest=[x for x in contacts[e] if end-2<=x['step']<=end]
 rows.append({'lane':e,'end':end,'termination_reason':meta['termination_reason_by_environment'][e],'status_count':len(status[e]),'expected_status_count':max(0,min(end,1060)-779)*16,'mismatched_count':sum(x['required']!=x['active'] for x in status[e]),'last_three_step_status':last,'contacts_near_end':latest})
(A/'contact-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for x in rows:print(x['lane'],'end',x['end'],'reason',x['termination_reason'],'status',x['status_count'],'expected',x['expected_status_count'],'count_mismatch',x['mismatched_count'],'near_end_contacts',x['contacts_near_end'][:5])
