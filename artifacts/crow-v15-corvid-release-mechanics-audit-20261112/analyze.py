from pathlib import Path
import gzip,json,collections,statistics
p=Path('artifacts/crow-v15-corvid-release-mechanics-audit-20261112')
paths={'held':Path('artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired'),'driven':Path('artifacts/crow-v15-corvid-hallux-amplitude-screen-20261112/candidate-20261127-rebound')}
result={}
for label,path in paths.items():
 sub={};q={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);
    if 201<=t<=300:sub[e,t,k]=list(map(float,z[6:]))
   elif z[0]=='CROW_Q_SUBSTEP':
    t,k,e=map(int,z[1:4]);q[e,t,k]=list(map(float,z[4:]))
 assert len(sub)==len(q)==12800
 events=[];skipped=collections.Counter()
 for e in range(8):
  for side in range(2):
   samples=[]
   for t in range(201,301):
    for k in range(16):
     v=sub[e,t,k];toes=v[5+8*side:13+8*side];samples.append((t,k,v[2+side]>1e-7,sum(toes[:6])>1e-7,sum(toes[6:])>1e-7))
   active=[i for i,x in enumerate(samples) if x[2]];groups=[]
   for i in active:
    if not groups or i-groups[-1][-1]>3:groups.append([i])
    else:groups[-1].append(i)
   ordinal=0
   for g in groups:
    first,last=g[0],g[-1]
    if first==0 or last==len(samples)-1:skipped['partial']+=1;continue
    front=[i for i in range(first,last+1) if samples[i][3]];rear=[i for i in range(first,last+1) if samples[i][4]]
    if not front or not rear:skipped['missing_group']+=1;continue
    ordinal+=1;ft,rt=front[-1],rear[-1];t,k=samples[ft][:2];coord=q[e,t,k];j=14+side*3;toe=23+side*8
    events.append({'lane':e,'side':side,'ordinal':ordinal,'first_tk':samples[first][:2],'last_front_tk':[t,k],'last_rear_tk':samples[rt][:2],'front_since_stance_ms':(ft-first)*1.25,'rear_since_stance_ms':(rt-first)*1.25,'rear_minus_front_ms':(rt-ft)*1.25,'joint_angles_at_front_rad':{'hip':coord[j],'knee':coord[j+1],'ankle':coord[j+2],'front1_proximal':coord[toe],'front2_proximal':coord[toe+2],'front3_proximal':coord[toe+4],'rear_proximal':coord[toe+6],'rear_distal':coord[toe+7]}})
 result[label]={'events':events,'skipped':dict(skipped)}
a={(x['lane'],x['side'],x['ordinal']):x for x in result['held']['events']};b={(x['lane'],x['side'],x['ordinal']):x for x in result['driven']['events']};keys=sorted(a.keys()&b.keys());pairs=[]
for key in keys:
 x,y=a[key],b[key];pairs.append({'key':key,'held_rear_minus_front_ms':x['rear_minus_front_ms'],'driven_rear_minus_front_ms':y['rear_minus_front_ms'],'front_stance_duration_change_ms':y['front_since_stance_ms']-x['front_since_stance_ms'],'rear_stance_duration_change_ms':y['rear_since_stance_ms']-x['rear_since_stance_ms'],'front_contact_joint_change_rad':{j:y['joint_angles_at_front_rad'][j]-x['joint_angles_at_front_rad'][j] for j in x['joint_angles_at_front_rad']}})
def stat(vals):
 return {'min':min(vals),'median':statistics.median(vals),'max':max(vals)} if vals else None
summ={'matched_ordinal_pairs':len(keys),'unmatched_held':len(a.keys()-b.keys()),'unmatched_driven':len(b.keys()-a.keys()),'held_rear_minus_front_ms':stat([x['rear_minus_front_ms'] for x in a.values()]),'driven_rear_minus_front_ms':stat([x['rear_minus_front_ms'] for x in b.values()]),'front_stance_duration_change_ms':stat([x['front_stance_duration_change_ms'] for x in pairs]),'rear_stance_duration_change_ms':stat([x['rear_stance_duration_change_ms'] for x in pairs]),'joint_change_rad':{j:stat([x['front_contact_joint_change_rad'][j] for x in pairs]) for j in ['hip','knee','ankle','front1_proximal','front2_proximal','front3_proximal','rear_proximal','rear_distal']}}
out={'summary':summ,'pairs':pairs,'arms':result,'scope':'Post-hoc different-controller comparison on one used seed and 2 s. Pairing by stance ordinal does not guarantee corresponding motions, especially when runs without toe groups are excluded. Joint samples at each trajectory own last-front-contact; these are associations, not causal effects or biological measurements. Stance boundaries merge gaps up to two 1.25 ms substeps; 1e-7 contact threshold; anatomical identity remains unverified.'};(p/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summ,indent=2));print('skipped',[(k,v['skipped']) for k,v in result.items()])
