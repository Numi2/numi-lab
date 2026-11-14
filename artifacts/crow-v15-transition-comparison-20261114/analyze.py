from pathlib import Path
import gzip,json,hashlib
R=Path('artifacts');A=R/'crow-v15-transition-comparison-20261114';folders={'near':'crow-v15-near-wing-transition-20261113','earlyphase':'crow-v15-earlier-phase-transition-20261114','latephase':'crow-v15-post-hop-phase-transition-20261114'};rows=[];hashes={}
for name,folder in folders.items():
 p=R/folder/'push-window.json.gz';hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.load(gzip.open(p,'rt'));s={(x['lane'],x['step']):x for x in d['state']};c={tuple(x[:3]):x[3:] for x in d['contact']};a={tuple(x[:3]):x[3:] for x in d['aero']}
 for e in (6,7):
  for t in range(369,396):
   if (e,t) not in s or not all((e,t,k) in c and (e,t,k) in a for k in range(16)):continue
   u=s[e,t];av=[a[e,t,k][3]*.02/16 for k in range(16)];rows.append({'trial':name,'lane':e,'interval':t,'observer_root_height_m':u['q'][2],'observer_vertical_velocity_m_s':u['v'][2],'observer_flap_q_rad':[u['q'][9],u['q'][10]],'foot_load_sum_Ns':sum(sum(c[e,t,k][2:4]) for k in range(16)),'positive_aero_vertical_Ns':sum(v for v in av if v>0),'negative_aero_vertical_Ns':sum(v for v in av if v<0)})
out={'source_sha256':hashes,'rows':rows,'scope':'used-seed association only; observer state epoch offset inferred for candidate lanes'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('rows',len(rows),'hashes',hashes)
