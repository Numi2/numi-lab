from pathlib import Path
import json,gzip
A=Path('artifacts/crow-v15-standing-bounce-audit-20261112');src=Path('artifacts/crow-v15-rear-toe-matched-trial-20261112/candidate-20261127-rebound-control/replay.json.gz');masses={}
for l in (A/'model.log').read_text().splitlines():
 z=l.split()
 if z[0]=='mass':masses[z[1]]=float(z[2])
with gzip.open(src,'rt') as f:r=json.load(f)['payload']
assert r['environment']==5 and r['body_state_stride']==13 and len(masses)==35
names=r['body_names'];frames={x['step']+1:x for x in r['frames'] if 1<=x['step']<=149};a=json.loads((A/'analysis.json').read_text())['rows'][5]['steps'];poserr=velerr=0.;n=0
for x in a:
 t=x['step'];
 if t not in frames:continue
 f=frames[t];bs=f['body_states'];total=sum(masses.values());zp=sum(m*bs[names.index(name)*13+2] for name,m in masses.items())/total;zv=sum(m*bs[names.index(name)*13+9] for name,m in masses.items())/total;poserr=max(poserr,abs(zp-x['com_z_m']));velerr=max(velerr,abs(zv-x['com_vz_m_s']));n+=1
out={'matched_frames':n,'com_position_max_error_m':poserr,'com_vertical_velocity_max_error_m_s':velerr,'scope':'Lane-5 replay compared to host reconstructed model mass-weighted body positions and linear velocities. Does not verify the source of collision impulses or scene material at substeps.'};(A/'replay-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert n==149 and poserr<1e-5 and velerr<1e-4
