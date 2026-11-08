import datetime, hashlib, subprocess, tarfile
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path('/Users/numi/numi-lab-nonempty-upload-build-20261108')
if (d/'preflight.sha256-check').exists():raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit('preflight failed')
status={}
for owner,storage,kind in [('baseline','private','control'),('baseline','private','retry'),('candidate','private','control'),('candidate','private','retry')]:
 name=f'{owner}-{storage}-{kind}';start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(name+'.stdout')).open('w') as out,(d/(name+'.stderr')).open('w') as err:
   code=str(subprocess.run([str(b/('probe-'+owner)),str(b/'MetalRobo.metallib'),str(d/(name+'.bin')),kind,storage],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
 except subprocess.TimeoutExpired:code='timeout'
 end=datetime.datetime.now(datetime.timezone.utc);status[name]=code
 (d/(name+'.timing')).write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={code}\n')
 print(f'{name} exit={code} start={start.isoformat()} end={end.isoformat()}',flush=True)
 if code!='0':break
 if kind=='retry' and 'rejected_status=' not in (d/(name+'.stdout')).read_text():break
pairs=[('baseline-private-control','baseline-private-retry'),('baseline-private-control','candidate-private-control'),('candidate-private-control','candidate-private-retry')]
lines=[]
for a,c in pairs:
 pa=d/(a+'.bin');pc=d/(c+'.bin')
 if status.get(a)=='0' and status.get(c)=='0' and pa.exists() and pc.exists():
  x=pa.read_bytes();y=pc.read_bytes();lines.append(f'{a} vs {c} bytes={len(x)},{len(y)} exact={x==y} sha256={hashlib.sha256(x).hexdigest()},{hashlib.sha256(y).hexdigest()}')
 else:lines.append(f'{a} vs {c} unavailable')
(d/'comparison.txt').write_text('\n'.join(lines)+'\n')
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/'raw.tar.gz','w:gz') as tar:
 for n in status:
  p=d/(n+'.bin')
  if p.exists():tar.add(p,arcname=p.name)
print(f'postflight={post} statuses={status}',flush=True)
