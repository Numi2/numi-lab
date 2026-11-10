import subprocess,hashlib,gzip,json
from pathlib import Path
R=Path.cwd();clone='/tmp/crow-restore-check-20261110';branch='agent/american-crow-rl-20261107'
subprocess.run(['git','-C',clone,'fetch','--quiet','--depth=1','origin',branch],check=True)
commit=subprocess.check_output(['git','-C',clone,'rev-parse','FETCH_HEAD'],text=True).strip(); rows=[]
for name in ['crow-v15-support-debounced-landing-20261110','crow-v15-narrow-perch-approach-20261110','crow-v15-perch-trim-20261110','crow-v15-toe-contact-reflex-20261110','crow-v15-toe-height-bracket-20261110']:
 for course in ['i','j']:
  for ext in ['json','tsv']:
   rel=f'artifacts/{name}/development-reserved-final-{course}.{ext}.gz';raw=R/f'build/{name}/development-reserved-final-{course}.{ext}'
   if not raw.exists():continue
   proc=subprocess.Popen(['git','-C',clone,'show',commit+':'+rel],stdout=subprocess.PIPE);a=hashlib.sha256();b=hashlib.sha256()
   with gzip.GzipFile(fileobj=proc.stdout) as z:
    while chunk:=z.read(1048576):a.update(chunk)
   if proc.wait()!=0:raise SystemExit('Git restore failed '+rel)
   with raw.open('rb') as f:
    while chunk:=f.read(1048576):b.update(chunk)
   if a.digest()!=b.digest():raise SystemExit('Restore mismatch '+rel)
   rows.append({'archive':rel,'raw':str(raw.relative_to(R)),'sha256':a.hexdigest(),'bytes':raw.stat().st_size})
   print('restored and matched',rel,flush=True)
manifest={'remote':'git@github.com:Numi2/numi-lab.git','branch':branch,'commit':commit,'method':'independent filtered clone; decompress git blob and compare SHA256 to original raw output','verified':rows}
(R/'artifacts/crow-v15-verified-restores-round2-20261110/restore.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in rows:(R/row['raw']).unlink()
print('retired verified duplicate bytes',sum(x['bytes'] for x in rows),flush=True)
