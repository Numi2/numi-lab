from pathlib import Path
import json,math
A=Path(__file__).resolve().parent;expected=json.loads((A/'expected-poses.json').read_text())['poses'];base={};data={};sha=None
for line in (A/'results-with-hash.tsv').read_text().splitlines():
 z=line.split()
 if z[0] not in ('B','C'):
  assert sha is None;sha=z[0];continue
 t=int(z[2])
 if z[0]=='B':base[t,z[3]]=list(map(float,z[4:]));continue
 h,k,a,g,cx,cy,cz,x,y,zz=map(float,z[3:]);data.setdefault(t,{})[h,k,a]={'foot_xyz_m':[x,y,zz],'com_xyz_m':[cx,cy,cz],'min_foot_and_toes_gap_m':g}
assert sha=='2f04b4345f2092777783f4523cc373fbd6d6862816e77843e4ed2ef31aef757f'
errs=[abs(x-y) for (t,name),p in base.items() for x,y in zip(p,expected[str(t)][name])];assert len(errs)==45*7
rows={}
for t,allsets in sorted(data.items()):
 basefoot=base[t,'crow_left_foot'][:3];choices={}
 for k in ((0.,0.,0.),(-.15,0.,0.),(.15,0.,0.),(0.,-.15,0.),(0.,.15,0.),(0.,0.,-.15),(0.,0.,.15)):
  if k in allsets:
   x=allsets[k];choices[str(k)]={'foot_delta_xyz_m':[a-b for a,b in zip(x['foot_xyz_m'],basefoot)],**x}
 rows[str(t)]={'native_foot_xyz_m':basefoot,'candidate_count':len(allsets),'isolated_offsets_rad':choices}
r={'verified_executable_sha256':sha,'max_native_pose_abs_error':max(errs),'native_pose_components':len(errs),'ticks':rows,'scope':'fixed-state FK of the collision model; not dynamically realized contact, grip or actuator tracking'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
