from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-common-flight-role-20261114');x=json.load(gzip.open(A/'launch-window.json.gz','rt'));s={(v['lane'],v['step']):v for v in x['state']};a={tuple(v[:3]):v[3:] for v in x['aero']};target={(v[0],v[1]):v[2:] for v in x['target']};rows=[]
for e in range(8):
 vals=[a[e,t,k][3]*.02/16 for t in range(371,378) for k in range(16) if (e,t,k) in a];samples=[]
 for t in (370,375,376,377,378,380):
  q,v=s[e,t]['q'],s[e,t]['v'];samples.append({'step':t,'height_m':q[2],'vertical_speed_m_s':v[2],'flap_q_rad':q[9:11],'flap_v_rad_s':v[8:10],'wing_targets':target.get((e,t),[])[8:12]})
 rows.append({'lane':e,'aero_substeps':len(vals),'expected_aero_substeps':112,'net_vertical_aero_impulse_Ns':sum(vals) if len(vals)==112 else None,'samples':samples})
out={'rows':rows,'scope':'post-hoc common-role launch comparison; measured modeled force, not evidence of biological causality; state/contact epoch alignment remains unverified'};(A/'launch-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'lane':r['lane'],'aero':r['net_vertical_aero_impulse_Ns'],'samples':[{'step':z['step'],'z':round(z['height_m'],4),'vz':round(z['vertical_speed_m_s'],4)} for z in r['samples']]} for r in rows],indent=2))
