from pathlib import Path
import subprocess,hashlib,zlib,json
root=Path.cwd();git='/tmp/crow-restore-check-20261110';items=['crow-v15-landing-root-balance-latch-screen-20261111','crow-v15-perch-hip-feedback-repeat-20261111'];commit=subprocess.check_output(['git','-C',git,'rev-parse','FETCH_HEAD'],text=True).strip();rows=[]
for name in items:
 manifest=subprocess.check_output(['git','-C',git,'show',f'{commit}:artifacts/{name}/SHA256SUMS'],text=True)
 expected=next(line.split()[0] for line in manifest.splitlines() if line.endswith('  trial/replay.json.gz'))
 p=subprocess.Popen(['git','-C',git,'show',f'{commit}:artifacts/{name}/trial/replay.json.gz'],stdout=subprocess.PIPE)
 compressed=hashlib.sha256();raw=hashlib.sha256();dec=zlib.decompressobj(31)
 while True:
  x=p.stdout.read(1024*1024)
  if not x:break
  compressed.update(x);raw.update(dec.decompress(x))
 raw.update(dec.flush());rc=p.wait();assert rc==0 and dec.eof
 local=root/'build'/name/'replay.json';lh=hashlib.sha256()
 with local.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):lh.update(x)
 row={'raw_path':str(local.relative_to(root)),'remote_commit':commit,'remote_compressed_sha256':compressed.hexdigest(),'remote_manifest_sha256':expected,'restored_raw_sha256':raw.hexdigest(),'local_raw_sha256':lh.hexdigest(),'remote_gzip_complete':dec.eof,'matches':expected==compressed.hexdigest() and raw.hexdigest()==lh.hexdigest()}
 rows.append(row);print(json.dumps(row),flush=True)
 assert row['matches']
(root/'artifacts/crow-evidence-retirement-replays-20261111/verification.json').write_text(json.dumps(rows,indent=2)+'\n')
