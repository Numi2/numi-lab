import json,math
from pathlib import Path
DT=.001

def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def mul(a,b):return [x*b for x in a]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def momentum(v,rv,masses):
 out=v[:3]
 for mass,r in zip(masses,rv):out=add(out,mul(r,mass))
 return out

def analyze(d):
 masses=d['rod_masses'];qprev=d['initial_q'];vprev=d['initial_v'];rprev=d['initial_rod_v'];totalmass=1+sum(masses);rows=[]
 for i,s in enumerate(d['steps']):
  accepted=s['world_code']==0 and s['code']==0;ground=[0.,0.,0.];jn=0.;wn=0.;wt=0.;rod=0;rigid=0;unknown=[]
  for c in s['rows']:
   if c['flags']&8:rod+=1;continue
   if c['body_a']!=0 or c['body_b']!=1:unknown.append([c['body_a'],c['body_b'],c['flags']]);continue
   rigid+=1;n=c['normal'];t=c['tangent'];b=cross(n,t);j=c['impulse'];normal=mul(n,j[0]);tangent=add(mul(t,j[1]),mul(b,j[2]));ground=add(ground,add(normal,tangent));jn+=j[0]
   lever=sub(c['point'],qprev[:3]);before=add(vprev[:3],cross(vprev[3:6],lever));after=add(s['v'][:3],cross(s['v'][3:6],lever));average=mul(add(before,after),.5);wn+=dot(normal,average);wt+=dot(tangent,average)
  dp=sub(momentum(s['v'],s['rod_v'],masses),momentum(vprev,rprev,masses));external=[0.,-9.81*totalmass*DT,0.];balance=sub(sub(dp,external),ground)
  rows.append({'step':i,'accepted':accepted,'rod_contacts':rod,'rigid_ground_contacts':rigid,'unknown_rigid_contacts':unknown,'ground_normal_impulse_Ns':jn,'ground_impulse_Ns':ground,'ground_normal_work_est_J':wn,'ground_tangent_work_est_J':wt,'momentum_balance_Ns':balance,'normal_residual_m_s':s['normal_residual'],'sweeps':s['sweeps']})
  if accepted:qprev=s['q'];vprev=s['v'];rprev=s['rod_v']
 active=[r for r in rows if r['accepted']];complete=len(active)==20
 bins=[sum(r['ground_normal_impulse_Ns'] for r in rows[j:j+5])/0.005 for j in range(0,20,5)] if complete else []
 return {'published':d['published'],'success_count':d['success'],'failed_count':d['failed'],'callback_noninvasive':d['capture_noninvasive'],'replay':d['repeat'],'trace_replay':d['exact_trace'],'sustained_mixed_steps':sum(r['rod_contacts']>0 and r['rigid_ground_contacts']>0 for r in active),'max_normal_residual_m_s':max((r['normal_residual_m_s'] for r in active),default=0.),'max_momentum_balance_Ns':max((max(map(abs,r['momentum_balance_Ns'])) for r in active if not r['unknown_rigid_contacts']),default=0.),'unknown_external_contact_steps':sum(bool(r['unknown_rigid_contacts']) for r in active),'force_bins_N':bins,'total_ground_impulse_Ns':sum(r['ground_normal_impulse_Ns'] for r in active),'total_ground_normal_work_est_J':sum(r['ground_normal_work_est_J'] for r in active),'total_ground_tangent_work_est_J':sum(r['ground_tangent_work_est_J'] for r in active),'steps':rows}

names=('fixed64','bounded','cap2');data={n:json.loads(Path(n+'.json').read_text()) for n in names};result={n:analyze(d) for n,d in data.items()}
ref=result['fixed64'];candidate=result['bounded'];comparison={'defined':False}
if len(ref['force_bins_N'])==len(candidate['force_bins_N'])==4 and ref['total_ground_impulse_Ns']>0:
 peak=max(abs(x) for x in ref['force_bins_N']);error=max(abs(a-b) for a,b in zip(ref['force_bins_N'],candidate['force_bins_N']))/peak
 impulse=abs(candidate['total_ground_impulse_Ns']-ref['total_ground_impulse_Ns'])/ref['total_ground_impulse_Ns']
 comparison={'defined':True,'force_bin_relative_peak':error,'impulse_relative':impulse,'both_below_0_25_percent':error<=.0025 and impulse<=.0025}
result['comparison']=comparison
Path('analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({n:({k:v for k,v in result[n].items() if k!='steps'} if n!='comparison' else result[n]) for n in result},indent=2))
