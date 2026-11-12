from pathlib import Path
import gzip,json,collections
p=Path('artifacts/crow-v15-corvid-front-stance-combination-20261112');old=Path('artifacts/crow-v15-corvid-front-support-screen-20261112/candidate-20261127-rebound');new=p/'candidate-20261127-rebound'
def read(path):
 st={};su={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);su[e,t,k]=z[4:]
 return st,su
a,b=read(new);c,d=read(old)
parity={'lane6_endpoint_mismatches':sum(a[6,t]!=c[6,t] for t in range(1,301)),'lane6_substep_mismatches':sum(b[6,t,k]!=d[6,t,k] for t in range(1,301) for k in range(16)),'all_lane_prewalk_endpoint_mismatches':sum(a[e,t]!=c[e,t] for e in range(8) for t in range(1,201)),'all_lane_prewalk_substep_mismatches':sum(b[e,t,k]!=d[e,t,k] for e in range(8) for t in range(1,201) for k in range(16))}
contact=json.loads((p/'contact-order.json').read_text())['own_or_opposite_candidate'];events=contact['events'];rows=[]
for e,amp in enumerate([.10,.20,.30,.35,.40,.50,.25,.60]):
 ev=[x for x in events if x['lane']==e];row={'lane':e,'stance_gain':amp,'complete_both_toe_group_runs':len(ev),'rear_first':sum(x['toeoff']=='rear_first' for x in ev),'front_first':sum(x['toeoff']=='front_first' for x in ev),'both_groups_by_side':dict(collections.Counter(x['side'] for x in ev))};rows.append(row)
out={'parity':parity,'lanes':rows,'counts':contact['counts'],'scope':'Development sweep uses one distinct start per amplitude; no independent stance gain effect is established. Stances lacking either toe group cannot determine release order. Full eligibility also requires clearance, support, slip and replication.'};(p/'sweep-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
