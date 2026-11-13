from pathlib import Path
import os,subprocess,json,runpy,hashlib
D=Path('artifacts/crow-v15-bank18-yaw-moment-audit-20261113');g=runpy.run_path(str(D/'analyze.py'));keys=g['keys']; exe=Path('build/crow-v15-bank-yaw-moment-audit-20261113/damping');lib=Path('build/crow-v15-mirrored-leg-isolation-20261113');res=subprocess.run([str(exe)],input=g['inputs'],text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.resolve())});vals={6:[],7:[]}
for l in res.stdout.splitlines():
 z=l.split(); assert z[0]=='D'; i=int(z[1]); vals[keys[i][0]].append(float(z[3]))
rows=[]
for lane,a in vals.items():
 assert len(a)==8016
 rows.append({'lane':lane,'mean_damping_yaw_moment_nm':sum(a)/len(a),'integrated_damping_yaw_impulse_nms':sum(a[:-1])*.00125})
out={'rows':rows,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'scope':'Posthoc FK estimate using model body damping coefficients and CPU damping convention; assumes body damping enabled and is not yet validated against GPU kernel or complete external-wrench balance.'};(D/'damping-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
