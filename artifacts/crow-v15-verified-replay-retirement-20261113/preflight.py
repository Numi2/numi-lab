from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107')
cases=[('crow-v15-mirrored-leg-wing-longflight-20261113','signed-turn-20261126-rebound-longflight','698883015c032df1ab91bc5ce5ddfeeff0e650de'),('crow-v15-mirrored-leg-wing-longflight-20261113','signed-turn-20261127-rebound-repeat-longflight','76ab0c15086ba3c042db8bffd91b4522fabbe162'),('crow-v15-mirrored-height-switch-descent-20261113','signed-turn-20261126-rebound-descent','dc5d376ff972e7a094b2577cbb61ac1974f8d5ce'),('crow-v15-leg-flight-height-switch-20261113','signed-turn-20261126-rebound-leg-switch','b6f4bf4ee224dd95fd3023452c8b8cafd2f51228'),('crow-v15-leg-flight-height-switch-20261113','signed-turn-20261127-rebound-repeat-leg-switch','f2b33e5adaeef1054865dd8340211df49bee42bb'),('crow-v15-bank-damping-sensitivity-20261113','signed-turn-20261126-rebound-damping','c178cca9a1b9ecd5a0881b9f97f06b1958173a36')]
rows=[]
for project,tag,commit in cases:
 for kind in ('trace.tsv','replay.json'):
  p=R/'build'/project/(tag+'-'+kind);assert p.is_file() and not p.is_symlink();s=p.stat();assert s.st_nlink==1
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  lsof=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True);assert lsof.returncode==1 and not lsof.stdout
  rows.append({'path':str(p),'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'bytes':s.st_size,'sha256':h.hexdigest(),'commit':commit,'github_path':'artifacts/'+project+'/'+tag+'/'+kind+'.gz','lsof_clear':True})
print(json.dumps({'rows':rows,'scope':'Read-only preflight; originals are not yet retired'},indent=2))
