from pathlib import Path
import gzip,collections,json,math,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');B=R/'crow-v15-supported-wing-transition-20261113';A=R/'crow-v15-supported-push-ground-impulse-audit-20261113';A.mkdir(exist_ok=True)
rows=[];logs={}
for arm in ('push','hold'):
 p=B/(arm+'-wing-20261126-rebound-supportedwing/run.stderr.gz');logs[arm]=hashlib.sha256(p.read_bytes()).hexdigest();d=collections.defaultdict(lambda:collections.defaultdict(float))
 with gzip.open(p,'rt') as f:
  for l in f:
   z=l.split()
   if not z or z[0]!='CROW_GROUND_CONTACT':continue
   t,k,e,a,b=map(int,z[1:6]);
   if e not in (2,3,4,7) or t<351 or t>355:continue
   body=a if b==35 else b
   typ='front' if body in list(range(19,25))+list(range(27,33)) else 'rear' if body in (25,26,33,34) else 'other'
   vals=list(map(float,z[8:]));jn,jt,jv=vals[12:15];assert jn>0
   x=d[e,typ];x['normal_ns']+=jn;x['tangential_magnitude_sum_ns']+=math.hypot(jt,jv);x['points']+=1;x['sliding_cone_near']+=int(math.hypot(jt,jv)>=.59*jn)
 for (e,typ),v in sorted(d.items()):rows.append({'arm':arm,'lane':e,'class':typ,**v,'tangent_to_normal_sum_ratio':v['tangential_magnitude_sum_ns']/v['normal_ns']})
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc sum of magnitudes of logged model contact-point impulses in ticks 351–355, not resultant net impulse and not measured slip; points may be speculative or warm-started, and distribution does not establish causation.','logs_sha256':logs,'rows':rows},indent=2)+'\n')
for x in rows:print(x['arm'],x['lane'],x['class'],'N',round(x['normal_ns'],4),'Tsum/N',round(x['tangent_to_normal_sum_ratio'],3),'near',int(x['sliding_cone_near']), '/',int(x['points']))
