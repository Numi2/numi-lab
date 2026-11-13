from pathlib import Path
import os,subprocess,json,runpy,hashlib
D=Path('artifacts/crow-v15-bank-yaw-moment-audit-20261113');g=runpy.run_path(str(D/'analyze.py'));keys=g['keys'];exe=Path('build/crow-v15-bank-yaw-moment-audit-20261113/damping-breakdown');lib=Path('build/crow-v15-mirrored-leg-isolation-20261113');p=subprocess.run([str(exe)],input=g['inputs'],text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.resolve())});vals={6:[],7:[]}
for line in p.stdout.splitlines():
 z=line.split();assert z[0]=='D';i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==36 and abs(sum(v[1:])-v[0])<1e-9;vals[keys[i][0]].append(v[1:])
rows=[]
for lane,a in vals.items():
 assert len(a)==8016
 sums=[sum(v[i] for v in a[:-1])*.00125 for i in range(35)];rows.append({'lane':lane,'body_yaw_damping_impulses_nms':sums,'torso_fraction_of_total':sums[0]/sum(sums)})
out={'rows':rows,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Posthoc FK body-by-body damping estimate assuming unit GPU scales. Includes translational-force moment arms and angular damping; actual GPU material parameters not yet confirmed.'};(D/'damping-breakdown.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
