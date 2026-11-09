#!/usr/bin/env python3
"""Download and restore one opaque archived fragment into a new destination.

The result is exact stored bytes, not an assertion that it is a valid Git pack.
The caller must supply the matching source manifest and a NEW output path.
"""
import hashlib,json,os,pathlib,shutil,stat,subprocess,sys,time
if len(sys.argv)!=3:raise SystemExit('usage: restore_fragment.py manifest.json NEW_DESTINATION')
manifest=pathlib.Path(sys.argv[1]); dest=pathlib.Path(sys.argv[2]).absolute()
m=json.loads(manifest.read_text())
if dest.exists():raise SystemExit('destination already exists')
if m['archive_repo']!='Numi2/workstation-checkpoint-archive':raise SystemExit('unexpected archive repo')
if pathlib.PurePath(m['asset']).name!=m['asset']:raise SystemExit('unsafe asset name')
if not dest.parent.is_dir():raise SystemExit('destination parent must exist')
floor=1024**3
if shutil.disk_usage(dest.parent).free < m['bytes']+floor:raise SystemExit('insufficient restore space above 1 GiB reserve')
cmd=['gh','release','download',m['release'],'--repo',m['archive_repo'],'--pattern',m['asset'],'--output',str(dest)]
proc=subprocess.Popen(cmd)
reason=None
while proc.poll() is None:
 if shutil.disk_usage(dest.parent).free<floor:
  reason='free-space reserve';proc.terminate();break
 time.sleep(2)
code=proc.wait()
if code or reason:
 if dest.exists():dest.unlink()
 raise SystemExit('download did not complete: '+str(reason or code))
h=hashlib.sha256()
with dest.open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
if dest.stat().st_size!=m['bytes'] or h.hexdigest()!=m['sha256']:
 raise SystemExit('download byte verification failed; destination retained for diagnosis')
os.chmod(dest,m['mode']);os.utime(dest,ns=(m['mtime_ns'],m['mtime_ns']))
s=dest.stat()
if stat.S_IMODE(s.st_mode)!=m['mode'] or s.st_mtime_ns!=m['mtime_ns']:
 raise SystemExit('metadata restoration failed')
print(json.dumps({'status':'PASS_EXACT_INDEPENDENT_RESTORE','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'asset':m['asset'],'bytes':s.st_size,'sha256':h.hexdigest(),'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'destination':str(dest),'pack_validity':'not established'},indent=2))
