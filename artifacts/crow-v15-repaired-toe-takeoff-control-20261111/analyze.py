from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-repaired-toe-takeoff-control-20261111')
def read(arm):
 sup={};sub={}
 with gzip.open(A/(arm+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=270:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=270:sub[lane,t,k]=list(map(float,v[6:]))
 assert len(sup)==8*270 and len(sub)==8*270*16,(len(sup),len(sub))
 return sup,sub
c,cs=read('control');p,ps=read('push');difference=[]
for lane in range(8):
 for t in range(1,248):
  for f in ('q','v'):
   if c[lane,t][f]!=p[lane,t][f]:difference.append([lane,t,f])
assert not difference,difference[:10]
def longest(xs):
 best=cur=0
 for x in xs:cur=cur+1 if x else 0;best=max(best,cur)
 return best
arms={}
for name,sup,sub in (('control',c,cs),('push',p,ps)):
 stdout=json.loads((A/(name+'.stdout')).read_text());rows=[]
 for lane in range(8):
  base=sup[lane,247]['q'][2];flags=[];up=[];normal=[];other=[]
  for t in range(248,271):
   z=[sub[lane,t,k] for k in range(16)];ints=[sum(q[i] for q in z) for i in range(7)];normal.append(ints[2]+ints[3]);up.append(ints[0]+ints[1]);other.append(ints[6]);flags.append(ints[2]+ints[3]<=1e-7 and sup[lane,t]['v'][2]>.05)
  rows.append({'lane':lane,'unloaded_upward_samples':sum(flags),'longest_unloaded_upward_run':longest(flags),'first_unloaded_upward_step':next((248+i for i,x in enumerate(flags) if x),None),'maximum_height_gain_m':max(sup[lane,t]['q'][2]-base for t in range(248,271)),'max_upward_speed_m_s':max(sup[lane,t]['v'][2] for t in range(248,271)),'total_foot_normal_impulse_Ns':sum(normal),'total_foot_upward_impulse_Ns':sum(up),'nonfoot_contact_intervals':sum(x>1e-7 for x in other),'passes_leg_hop_gate':longest(flags)>=2 and all(x<=1e-7 for x in other)})
 arms[name]={'failed_environment_steps':stdout['failed_environment_steps'],'termination_count':stdout['termination_count'],'lanes':rows}
result={'preextension_qv_equal_through_step':247,'arms':arms,'component_gate':all(r['passes_leg_hop_gate'] for r in arms['push']['lanes']) and not any(r['passes_leg_hop_gate'] for r in arms['control']['lanes']) and all(v['failed_environment_steps']==0 and v['termination_count']==0 for v in arms.values()),'scope':'one development seed, eight perturbed starts, no active flapping in leg-only window but passive aerodynamics retained; no takeoff-to-flight or biological qualification'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
