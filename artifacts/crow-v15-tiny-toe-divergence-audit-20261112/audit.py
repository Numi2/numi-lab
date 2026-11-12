from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-tiny-toe-divergence-audit-20261112');P=Path('artifacts/crow-v15-tiny-swing-toe-screen-20261112');q={}
for n,d in [('sham','sham-20261127-dev2'),('candidate','candidate-20261127-rebound')]:
 states={};subs=collections.defaultdict(lambda: [0.0]*18)
 with gzip.open(P/d/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);
    if 210<=t<=300:
     vals=[float(z[8]),float(z[9])]+[float(v) for v in z[11:27]]
     assert len(vals)==18
     subs[e,t]=[a+b for a,b in zip(subs[e,t],vals)]
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if 1<=x['step']<=300:states[x['lane'],x['step']]=x
 assert len(states)==2400;q[n]=states;q[n+'_substeps']=subs
rows=[]
for e in range(8):
 first=next(t for t in range(201,301) if q['sham'][e,t]['q']!=q['candidate'][e,t]['q'] or q['sham'][e,t]['v']!=q['candidate'][e,t]['v']);x=q['sham'][e,first];y=q['candidate'][e,first];dq=[b-a for a,b in zip(x['q'],y['q'])];dv=[b-a for a,b in zip(x['v'],y['v'])];rows.append({'lane':e,'first_divergent_interval':first,'q_max_abs':max(map(abs,dq)),'v_max_abs':max(map(abs,dv)),'left_rear_prox_q_difference_rad':dq[29],'right_rear_prox_q_difference_rad':dq[37],'root_forward_velocity_difference_m_s':dv[0],'policy_outputs_equal_at_first_divergence':x['policy_action']==y['policy_action'],'step201_root_x_delta_m':q['candidate'][e,201]['q'][0]-q['sham'][e,201]['q'][0],'step250_root_x_delta_m':q['candidate'][e,250]['q'][0]-q['sham'][e,250]['q'][0],'step300_root_x_delta_m':q['candidate'][e,300]['q'][0]-q['sham'][e,300]['q'][0]})
def endpoint_contact(x,side):
 relevant=[c for c in x["contacts"] if c["a_name"].startswith("crow_"+side+"_") and c["b_name"]=="locomotion_ground"]
 return {"normal_impulse_sum":sum(abs(c["impulses"][0]) for c in relevant),"loaded_contacts":sum(abs(c["impulses"][0])>1e-8 for c in relevant),"rear_toe_normal_impulse_sum":sum(abs(c["impulses"][0]) for c in relevant if "rear_toe" in c["a_name"])}
contact_windows=[]
for e in range(8):
 for t in (214,215,216,220):
  for arm in ("sham","candidate"):
   contact_windows.append({"lane":e,"interval":t,"arm":arm,"left":endpoint_contact(q[arm][e,t],"left"),"right":endpoint_contact(q[arm][e,t],"right")})
substep_summaries=[]
for e in range(8):
 for t in (214,215,216,220,225,250,300):
  for arm in ('sham','candidate'):
   v=q[arm+'_substeps'][e,t];substep_summaries.append({'lane':e,'interval':t,'arm':arm,'left_foot_normal_sum':v[0],'right_foot_normal_sum':v[1],'left_rear_toe_normal_sum':sum(v[8:10]),'right_rear_toe_normal_sum':sum(v[16:18])})
out={'rows':rows,'substep_summaries':substep_summaries,'contact_windows':contact_windows,'scope':'Endpoint accepted state comparison of a contact-switched physical toe command and matched sham. First observed divergence is not proof of cause; feedback and contacts are coupled, applied toe torque and exact switch events are not logged.'};(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':rows,'example_contact_windows':[x for x in contact_windows if x['lane']==0]},indent=2))
