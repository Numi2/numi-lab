from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
if not rev.startswith('0b810ea1'):raise SystemExit('unexpected remote revision '+rev)
A='artifacts/crow-v15-matched-ccd-grip-probe-20261110/';items=[]
for name in ('smoke.txt','open-smoke.txt','interrupted-control-schedule/results.txt'):
 archive=A+name+'.gz';remote=subprocess.check_output(['git','-C',str(C),'show',rev+':'+archive]);local=(R/archive).read_bytes()
 if remote!=local:raise SystemExit('archive mismatch '+archive)
 restored=gzip.decompress(remote);raw=R/(A+name);raw_present=raw.exists()
 if raw_present and restored!=raw.read_bytes():raise SystemExit('raw mismatch '+name)
 items.append({'archive':archive,'sha256':hashlib.sha256(remote).hexdigest(),'restored_sha256':hashlib.sha256(restored).hexdigest(),'bytes':len(restored),'raw_present':raw_present})
for name in ('smoke.txt','open-smoke.txt'):(R/(A+name)).unlink()
(R/'artifacts/crow-v15-restoration-matched-smokes-20261110/verification.json').write_text(json.dumps({'remote_commit':rev,'items':items,'note':'Interrupted raw was already replaced by the corrected live run; independently verified the exact remote gzip against the retained local archive and restored it.'},indent=2)+'\n')
print('Verified remote archives; retired two smoke duplicates.')
