from pathlib import Path
import json,hashlib,subprocess,tarfile,shutil,os
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'old-mesh-preparation-308-recovery'
m=json.loads((D/'human-old-mesh-preparation-308.json').read_text())
plan=json.loads((W/'shared-bundle-native-307/install-motion.plan.json').read_text())
env=plan['stages'][0]['environment']
names=['ROLE_D_FULL86_CSV','ROLE_I_FULL86_CSV','ROLE_O_FULL86_CSV','TETGEN_ELEMENTS']
paths={env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'+k] for k in names}
rows={r['relative']:r for r in m['files'] if r['path'] in paths}
assert len(rows)==4 and all(not Path(r['path']).exists() for r in rows.values())
out=W/'baseline-repeat-332-restoration';out.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
subprocess.run(['/opt/homebrew/bin/gh','release','download','numi-20261104-cold-human-evidence','--repo','Numi2/macmini-evidence-archive','--pattern',m['asset'],'--dir',str(out)],check=True)
asset=out/m['asset'];assert sha(asset)==m['archive_sha256']
decoder=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','-q','-c',str(asset)],stdout=subprocess.PIPE);restored=[]
with tarfile.open(fileobj=decoder.stdout,mode='r|') as archive:
 for member in archive:
  if member.name not in rows:continue
  r=rows[member.name];assert member.isfile() and member.size==r['size']
  dest=Path(r['path']);partial=dest.with_suffix('.restoring332')
  with archive.extractfile(member) as a,partial.open('xb') as b:shutil.copyfileobj(a,b,1<<20)
  assert partial.stat().st_size==r['size'] and sha(partial)==r['sha256']
  os.chmod(partial,r['mode']);os.utime(partial,ns=(r['mtime_ns'],r['mtime_ns']))
  os.replace(partial,dest);restored.append(r)
decoder.stdout.close();assert decoder.wait()==0 and len(restored)==len(rows)
(out/'receipt.json').write_text(json.dumps({'status':'PASS_EXACT_BASELINE_INPUT_RESTORATION','archive_sha256':m['archive_sha256'],'files':restored},indent=2)+'\n')
asset.unlink();print('RESTORED_BASELINE_INPUTS',len(restored),'free_GiB',shutil.disk_usage(W).free/2**30,flush=True)
