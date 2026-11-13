from pathlib import Path
import json
A=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-leg-work-observer-20261113');push=json.loads((A/'analysis.json').read_text());hold=json.loads((A/'nopush-analysis.json').read_text());res=[]
for e in range(2,8):
 p=next(r for r in push['rows'] if r['lane']==e and r['steps']==[348,365]);h=next(r for r in hold['rows'] if r['lane']==e and r['steps']==[348,365]);res.append({'lane':e,'push_net_j':p['summed_net_j'],'hold_net_j':h['summed_net_j'],'net_difference_j':p['summed_net_j']-h['summed_net_j'],'push_positive_j':p['summed_positive_j'],'hold_positive_j':h['summed_positive_j'],'push_ground_impulse_ns':p['upward_foot_impulse_ns'],'hold_ground_impulse_ns':h['upward_foot_impulse_ns'],'impulse_difference_ns':p['upward_foot_impulse_ns']-h['upward_foot_impulse_ns']})
(A/'paired-comparison.json').write_text(json.dumps({'steps':[348,365],'rows':res},indent=2)+'\n')
for x in res:print('lane',x['lane'],'net-work push/hold',round(x['push_net_j'],6),round(x['hold_net_j'],6),'ground impulse difference',round(x['impulse_difference_ns'],6))
