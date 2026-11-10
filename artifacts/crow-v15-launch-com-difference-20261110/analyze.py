from pathlib import Path
import gzip,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-launch-com-difference-20261110';out={}
for label,rel in [('flight_enabled','artifacts/crow-v15-integrated-push-causality-20261110/diagnostic-reserved-final-j-push.stderr.gz'),('flap_held','artifacts/crow-v15-launch-wing-ablation-20261110/diagnostic-reserved-final-j-push.stderr.gz')]:
 d={}
 with gzip.open(R/rel,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);t=x['step']
    if 650<=t<=670:d[t]=x
 inp=''.join(str(t)+' '+' '.join(map(str,z['q']))+'\n' for t,z in sorted(d.items()));p=subprocess.run([str(R/'build/crow-v15-perch-force-observer-20261110/com_probe')],input=inp,text=True,capture_output=True,check=True)
 com={int(a[0]):list(map(float,a[2:])) for a in map(str.split,p.stdout.splitlines())};rows=[]
 for t in range(654,660):
  accel=[(com[t+1][j]-2*com[t][j]+com[t-1][j])/.02**2 for j in range(3)]
  rows.append({'step':t,'com':com[t],'central_difference_accel_m_s2':accel,'reported_endpoint_contacts':len(d[t]['contacts'])})
 out[label]=rows
(A/'analysis.json').write_text(json.dumps({'samples':out,'scope':'COM from native articulated geometry and accepted positions; central differences across three control endpoints. Combined external acceleration, not separated aerodynamic and other external forces; no per-substep attribution.'},indent=2)+'\n');print(json.dumps(out,indent=2))
