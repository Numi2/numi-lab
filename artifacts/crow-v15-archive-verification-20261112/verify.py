from pathlib import Path
import subprocess,hashlib,gzip,json
R=Path.cwd();C=Path('/tmp/crow-archive-verify-20261112');A=R/'artifacts/crow-v15-archive-verification-20261112';rev=subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip();assert rev=='750047a2956c538d9cdee973d762e0c6fac171a0'
names=['crow-v15-standing-target-feedback-matched-20261112','crow-v15-standing-walk-integration-20261112','crow-v15-toe-preload-dose-screen-20261112','crow-v15-toe-preload-matched-trial-20261112','crow-v15-toe-preload-second-seed-20261112','crow-v15-wing-bank-turn-screen-20261112','crow-v15-wing-load-observer-20261112','crow-v15-wing-root-moment-observer-20261112'];results=[]
for name in names:
 remote='artifacts/'+name
 manifest=subprocess.check_output(['git','-C',str(C),'show',rev+':'+remote+'/SHA256SUMS'],text=True)
 hashes={line.split('  ',1)[1]:line.split('  ',1)[0] for line in manifest.splitlines() if '  ' in line}
 for raw in sorted((R/'build'/name).glob('*')):
  if not raw.name.endswith(('-replay.json','-trace.tsv')):continue
  if raw.name.startswith('candidate-'):
   tag,kind=raw.name.rsplit('-',1);relative=tag+'/'+kind+'.gz'
  else:relative=raw.name+'.gz'
  assert relative in hashes,(name,relative)
  blob=subprocess.check_output(['git','-C',str(C),'show',rev+':'+remote+'/'+relative]);h=hashlib.sha256(blob).hexdigest();assert h==hashes[relative],(name,relative,'manifest')
  restored=gzip.decompress(blob);rawhash=hashlib.sha256(raw.read_bytes()).hexdigest();assert hashlib.sha256(restored).hexdigest()==rawhash,(name,relative,'raw')
  local=R/remote/relative;assert hashlib.sha256(local.read_bytes()).hexdigest()==h,(name,relative,'local')
  results.append({'raw':str(raw.relative_to(R)),'raw_sha256':rawhash,'raw_bytes':raw.stat().st_size,'archive':remote+'/'+relative,'compressed_sha256':h})
assert len(results)==22,len(results)
out={'remote_commit':rev,'files_verified':len(results),'total_raw_bytes':sum(x['raw_bytes'] for x in results),'files':results};(A/'verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'remote_commit':rev,'files_verified':len(results),'total_raw_bytes':out['total_raw_bytes']}))
