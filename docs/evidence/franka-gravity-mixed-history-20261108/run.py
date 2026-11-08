import datetime,hashlib,struct,subprocess,tarfile
from pathlib import Path
d=Path(__file__).resolve().parent
b=Path('/Users/numi/numi-franka-gravity-mixed-history-build-20261108')
if (d/'preflight.sha256-check').exists(): raise SystemExit('refusing overwrite')
with (d/'preflight.sha256-check').open('w') as f:
 if subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode: raise SystemExit('preflight failed')
status={}
for arm in ('control','mixed'):
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(arm+'.stdout')).open('w') as o,(d/(arm+'.stderr')).open('w') as e:
   code=str(subprocess.run([str(b/'probe'),str(b/'WorldRenderer.metallib'),str(b/'pick_object.mrvpack'),str(d/(arm+'.bin')),arm],cwd=d,stdout=o,stderr=e,timeout=120).returncode)
 except subprocess.TimeoutExpired: code='timeout'
 end=datetime.datetime.now(datetime.timezone.utc);status[arm]=code
 (d/(arm+'.timing')).write_text(f'start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={code}\n')
 print(f'{arm} exit={code} start={start.isoformat()} end={end.isoformat()}',flush=True)
 if code!='0': break
names=('depth_history','track_history','frame_depth','frame_id','frame_valid','camera','metadata','cadence','actor_history')
sizes=(4,4,4,4,4,4,4,8,1)
def read(path):
 data=path.read_bytes();p=0;out=[]
 while p<len(data):
  group={}
  for n,s in zip(names,sizes):
   if p+8>len(data):raise ValueError('short length')
   count=struct.unpack_from('<Q',data,p)[0];p+=8;end=p+count*s
   if end>len(data):raise ValueError('short field')
   group[n]=data[p:end];p=end
  out.append(group)
 return out
lines=[]
for arm in status:
 try:
  snapshots=read(d/(arm+'.bin'));lines.append(f'{arm} snapshots={len(snapshots)}')
  for i in range(1,len(snapshots)):
   for n in names:
    x=snapshots[i-1][n];y=snapshots[i][n]
    lines.append(f'{arm} previous_to_{i} {n} size={len(x)},{len(y)} equal={x==y} warm_sha256={hashlib.sha256(x).hexdigest()} next_sha256={hashlib.sha256(y).hexdigest()}')
    if n=='actor_history' and len(x)==len(y) and len(x)%2==0:
     h=len(x)//2
     for e in (0,1):lines.append(f'{arm} previous_to_{i} actor_env{e}_equal={x[e*h:(e+1)*h]==y[e*h:(e+1)*h]} bytes={h}')
 except (ValueError,OSError) as ex: lines.append(f'{arm} parse_error={ex}')
if status.get('control')=='0' and status.get('mixed')=='0':
 try:
  control=read(d/'control.bin');mixed=read(d/'mixed.bin')
  for i in range(min(len(control),len(mixed))):
   x=control[i]['actor_history'];y=mixed[i]['actor_history']
   if len(x)==len(y) and len(x)%2==0:
    h=len(x)//2;lines.append(f'control_vs_mixed step={i} actor_env1_equal={x[h:]==y[h:]} bytes={h}')
 except (ValueError,OSError) as ex:lines.append(f'cross_arm_parse_error={ex}')
(d/'comparison.txt').write_text('\n'.join(lines)+'\n')
with (d/'postflight.sha256-check').open('w') as f:post=subprocess.run(['shasum','-a','256','-c','declared-identities.sha256'],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/'raw.tar.gz','w:gz') as tar:
 for arm in status:
  p=d/(arm+'.bin')
  if p.exists():tar.add(p,arcname=p.name)
print(f'postflight={post} statuses={status}',flush=True)
