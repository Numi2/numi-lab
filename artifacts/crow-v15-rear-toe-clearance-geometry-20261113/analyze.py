from pathlib import Path
import json,math
A=Path('artifacts/crow-v15-rear-toe-clearance-geometry-20261113');expect=json.loads((A/'expected-poses.json').read_text())['poses'];base={};data={};hashes=[]
for line in (A/'results.tsv').read_text().splitlines():
 z=line.split()
 if z[0] not in ('B','C'):hashes.append(z[0]);continue
 t=int(z[2])
 if z[0]=='B':base[t,z[3]]=list(map(float,z[4:]));continue
 hip,ankle,rp,rd,gap,gp,gd,fx,fy,fz=map(float,z[3:]);data.setdefault(t,[]).append({'offsets_rad':[hip,ankle,rp,rd],'all_gap_m':gap,'rear_proximal_gap_m':gp,'rear_distal_gap_m':gd,'foot_xyz_m':[fx,fy,fz]})
assert len(hashes)==2 and hashes[1]=='2401d8c6aa048f7a4a75b51d2fbc1498b97b66fd3e3e6aeb19b3014fa5af7aea'
errs=[abs(a-b) for (t,n),v in base.items() for a,b in zip(v,expect[str(t)][n])];assert len(errs)==54*7
out={}
for t,items in sorted(data.items()):
 original=next(x for x in items if x['offsets_rad']==[0.,0.,0.,0.]);isolated=[x for x in items if x['offsets_rad'][0:2]==[0.,0.] and x['offsets_rad'][3]==0.];valid=[x for x in items if x['all_gap_m']>=.001];valid.sort(key=lambda x:(sum(a*a for a in x['offsets_rad']),-x['all_gap_m']));out[str(t)]={'candidates':len(items),'baseline':original,'isolated_rear_proximal_offsets':isolated,'at_least_1mm_count':len(valid),'smallest_norm_candidates':valid[:8]}
r={'source_hashes':hashes,'max_pose_abs_error':max(errs),'pose_components':len(errs),'ticks':out,'scope':'fixed-root geometry only, no contact, dynamics, controller or biological inference'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
