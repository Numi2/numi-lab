from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-hip-roll-original-flight-isolation-20261111');old=Path('artifacts/crow-v15-two-segment-frozen-flight-envelope-20261111/run.stderr.gz');cur={};ref={};endvel={};caps=collections.defaultdict(list);ground=collections.Counter();groundother=collections.Counter();counts=collections.Counter()
with gzip.open(old,'rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if x['lane']<5:ref[x['lane'],x['step']]=x
with gzip.open(A/'flight.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);cur[x['lane'],x['step']]=x
  else:
   a=l.split()
   if a and a[0]=='CROW_TOE_V_SUBSTEP':
    t,k,e=map(int,a[1:4]);vv=list(map(float,a[4:]));counts[e,t,k]+=1
    if max(map(abs,vv))>=11.99:caps[e].append((t,k))
    if k==15 and t%100==0:endvel[e,t]=vv
   elif a and a[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,a[1:4]);
    if t>=300:
     if abs(float(a[8]))+abs(float(a[9]))>1e-7:ground[e]+=1
     if abs(float(a[10]))>1e-7:groundother[e]+=1
rows=[]
for e in range(5):
 common=sorted(t for lane,t in cur.keys()&ref.keys() if lane==e);tail=[t for t in common if 300<=t<=1600];vd=[max(abs(a-b) for a,b in zip(cur[e,t]['v'][22:38],endvel[e,t])) for t in common if (e,t) in endvel];pos=[math.dist(cur[e,t]['q'][:3],ref[e,t]['q'][:3]) for t in tail]
 rows.append({'lane':e,'common_sample_steps':common,'first_sample_position_difference_m':math.dist(cur[e,1]['q'][:3],ref[e,1]['q'][:3]) if 1 in common else None,'max_position_difference_sampled_steps_300_1600_m':max(pos) if pos else None,'max_toe_velocity_observer_endpoint_difference_rad_s':max(vd) if vd else None,'endpoint_comparisons':len(vd),'toe_cap_count':len(caps[e]),'last_toe_cap_observer_step':caps[e][-1][0] if caps[e] else None,'toe_cap_count_at_or_after_300':sum(t>=300 for t,k in caps[e]),'ground_foot_normal_substeps_at_or_after_300':ground[e],'ground_other_normal_substeps_at_or_after_300':groundother[e],'record_count':sum(counts[e,t,k] for t in range(1,1602) for k in range(16)),'missing_or_duplicate_count':sum(counts[e,t,k]!=1 for t in range(1,1602) for k in range(16))})
out={'rows':rows,'limitations':'Five development lanes and sparse body-state samples; ground-contact counters use observed positive normal impulses. Does not establish continuous geometric clearance, perturbation recovery, full qualification, route or landing. Initial ground phase toe-speed cap hits remain failures under the full-run no-cap criterion.'};(A/'phase-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
