from pathlib import Path
import json,hashlib,datetime
import numpy as np
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native565-fine-perturbation-descriptive-001';d.mkdir(exist_ok=False)
cp=q/'solver/control19-newton3-small-epsilon-resume-diagnostic-565/diagnostics/capture-operator-fd-capture/capture';op=q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=[json.loads(l) for l in cp.read_text().splitlines()];old=[json.loads(l) for l in op.read_text().splitlines()]
h=a[0];ns=[r for r in a if r['record']=='node'];ts=[r for r in a if r['record']=='tetrahedron']
oldnodes={(r['environment'],r['node_id']):r for r in old if r['record']=='node'};oldtets={(r['environment'],r['tet_id']):r for r in old if r['record']=='tetrahedron'}
assert all(n['center']==oldnodes[n['environment'],n['node_id']]['center'] and n['correction_delta_v_m_s']==oldnodes[n['environment'],n['node_id']]['correction_delta_v_m_s'] for n in ns)
assert all(t['center']==oldtets[t['environment'],t['tet_id']]['center'] and t['cached_algorithmic_tangent_local']==oldtets[t['environment'],t['tet_id']]['cached_algorithmic_tangent_local'] for t in ts)
norm=lambda x:float(np.linalg.norm(x))
def comp(x,y):return {'absolute_l2':norm(x-y),'reference_l2':norm(y),'observed_l2':norm(x),'relative_l2':norm(x-y)/norm(y) if norm(y)>0 else None}
reports=[]
for env in range(h['environment_count']):
 n=sorted((r for r in ns if r['environment']==env),key=lambda r:r['node_id']);t=[r for r in ts if r['environment']==env];free=np.array([r['center']['constraint_tag'] not in (1,3) and r['center']['mass_kg']>0 for r in n]);dv=np.array([r['correction_delta_v_m_s'] for r in n]);action=np.array([r['production_A_delta_v_impulse_Ns'][:3] for r in n]);center=np.array([r['center_residual_impulse_Ns'][:3] for r in n]);dt=h['dt_seconds_by_environment'][env]
 report={'environment':env,'free_nodes':int(sum(free)),'action_l2_Ns':norm(action[free]),'epsilon_pairs':[]}
 for j,e in enumerate(h['epsilon_values']):
  rp=np.array([r['epsilon_pair_residual_impulses_Ns'][2*j][:3] for r in n]);rm=np.array([r['epsilon_pair_residual_impulses_Ns'][2*j+1][:3] for r in n]);central=-(rp-rm)/(2*e);plus=-(rp-center)/e;minus=-(center-rm)/e
  result={'epsilon':e,'central_action':comp(central[free],action[free]),'plus_action':comp(plus[free],action[free]),'minus_action':comp(minus[free],action[free]),'native_residual_pair_delta_l2_Ns':norm((rp-rm)[free]),'realized_nodal_relationships':[]}
  for sign,label,slot in [(1,'plus',2*j),(-1,'minus',2*j+1)]:
   realized_v=np.array([r['epsilon_pair_realized_velocity_delta_m_s'][slot] for r in n]);realized_u=np.array([r['epsilon_pair_realized_u_delta_m'][slot] for r in n]);realized_x=np.array([r['epsilon_pair_realized_position_delta_m'][slot] for r in n])
   result['realized_nodal_relationships'].append({'side':label,'velocity':comp(realized_v[free],sign*e*dv[free]),'reference_displacement':comp(realized_u[free],sign*e*dt*dv[free]),'world_position':comp(realized_x[free],sign*e*dt*dv[free]),'nonzero_velocity_components':int(np.count_nonzero(realized_v[free])),'nonzero_u_components':int(np.count_nonzero(realized_u[free])),'nonzero_x_components':int(np.count_nonzero(realized_x[free]))})
  f_actual=[];f_nominal=[]
  for r in t:
   p,m=r['sides'][2*j:2*j+2]
   pp=np.array(p['F_local'])+np.array(p['F_local_low']);mm=np.array(m['F_local'])+np.array(m['F_local_low'])
   f_actual.append((pp-mm)/(2*e*dt));f_nominal.append(np.array(r['center']['F_direction_local_per_s']))
  result['realized_local_F_direction']=comp(np.array(f_actual),np.array(f_nominal))
  report['epsilon_pairs'].append(result)
 reports.append(report)
r={'schema':'numi.cardboard.native565-fine-perturbation-descriptive.v1','capture_sha256':sha(cp),'coarse_capture_sha256':sha(op),'native_centers_corrections_cached_tangents_identical_to555':True,'reports':reports,'qualification':'INCONCLUSIVE_NO_EMPIRICAL_NOISE_FLOOR','limits':'Descriptive native finite differences and representability only. No independent projection reference or smooth-branch gate, no empirical absolute noise floor and no derivative pass; quiet zero-action environment remains nonqualifying.','physical_transactions_added':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'report_sha256':sha(d/'report.json'),'reports':reports}),flush=True)
