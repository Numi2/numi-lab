from pathlib import Path
import gzip,hashlib,json
A=Path('artifacts/crow-v15-launch-aero-impulse-audit-20261114');P=Path('artifacts/crow-v15-bounded-flight-tuck-20261114/push-window.json.gz');D=json.load(gzip.open(P,'rt'));rows=[]
for e in (0,3,7):
 for lo,hi in ((376,384),(384,392),(376,396)):
  a=[z for z in D['aero'] if z[0]==e and lo<=z[1]<hi];c=[z for z in D['contact'] if z[0]==e and lo<=z[1]<hi];assert len(a)==len(c)==(hi-lo)*16
  rows.append({'lane':e,'interval_start_inclusive':lo,'interval_end_exclusive':hi,'aero_samples':len(a),'vertical_aero_impulse_Ns':sum(z[6] for z in a)*.02/16,'gravity_impulse_Ns_assuming_mass_0_45006kg':.45006*9.81*(hi-lo)*.02,'upward_foot_contact_impulse_Ns':sum(sum(z[3:5]) for z in c)})
out={'source_raw_sha256':D['source_sha256'],'source_window_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rows':rows,'trace_schema':'observer.mm CROW_AERO_FORCE prints stride,total xyz; index 6 is total z','limitations':'Model force estimate and normal-only foot impulses, not biological measurements; contact/state/aero observer epochs are not independently synchronized and other forces are not a closed momentum balance.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
