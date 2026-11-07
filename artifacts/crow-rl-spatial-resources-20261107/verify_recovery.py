#!/usr/bin/env python3
"""Fresh-process native replay using only exact Git-restored runtime roles."""
from pathlib import Path
import hashlib,json,os,stat,subprocess,sys
root=Path.cwd();base=root/'artifacts/crow-rl-spatial-resources-20261107';tmp=Path('/private/tmp/crow-spatial-git-recovery-20261107')
plan={'prediction':'One Git-restored native foraging executable runs its full fixed admission from a fresh process. Every reference file must match byte for byte except summary wall_seconds, which must be finite/positive; all other summary fields match. Same hardware/OS/dependencies; no behavior or generalization claim.','budget':'One180second native replay; abort any mismatch, preserve failure. Verify all temporary file identities and open handles before exact-path cleanup.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n');command=[str(tmp/'bin/metalrobo_crow_foraging_check'),str(tmp/'replay')]
try:
 p=subprocess.run(command,capture_output=True,text=True,timeout=180)
 (base/'recovery.stdout').write_text(p.stdout);(base/'recovery.stderr').write_text(p.stderr)
 if p.returncode:raise RuntimeError('Git native replay failed: '+p.stderr[-3000:])
 reference=tmp/'reference';replay=tmp/'replay'
 names={str(p.relative_to(reference)) for p in reference.rglob('*') if p.is_file()}
 if names!={str(p.relative_to(replay)) for p in replay.rglob('*') if p.is_file()}:raise RuntimeError('Git replay file sets differ')
 matches={}
 for name in sorted(names):
  a=(reference/name).read_bytes();b=(replay/name).read_bytes()
  if name=='summary.json':
   old,new=json.loads(a),json.loads(b)
   for report in (old,new):
    seconds=report.pop('wall_seconds')
    if not isinstance(seconds,(int,float)) or not 0<seconds<180:raise RuntimeError('Invalid replay elapsed time')
   if old!=new or new.get('status')!='passed':raise RuntimeError('Git replay native report differs')
  else:
   if a!=b:raise RuntimeError('Git replay differs: '+name)
   matches[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 files=[];dirs=[]
 for directory,subdirs,leaves in os.walk(tmp,followlinks=False):
  for name in subdirs:
   d=Path(directory)/name
   if d.is_symlink():raise RuntimeError('Unexpected runtime directory symlink')
   dirs.append(d)
  for name in leaves:
   f=Path(directory)/name;s=f.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected runtime file type')
   files.append({'path':str(f),'bytes':s.st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovered runtime handles')
 before=subprocess.check_output(['df','-k','.'],text=True)
 for f in files:
  path=Path(f['path']);s=path.lstat()
  if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise RuntimeError('Restoration changed during cleanup')
 v={'status':'passed','archive_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'plan':plan,'command':command,'all_manifest_files_verified':294,'runtime_restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'exact_replay_files':matches,'native_report':json.loads(p.stdout),'temporary_removal':{'status':'validated','open_handles':0,'files':files,'df_before':before}}
 (base/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
 for f in files:Path(f['path']).unlink()
 for d in sorted(dirs,key=lambda p:len(p.parts),reverse=True):d.rmdir()
 tmp.rmdir();v['temporary_removal'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True))
 (base/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'status':'passed','native_replay_exact_files':len(matches),'temporary_files_removed':len(files),'temporary_bytes_removed':sum(f['bytes'] for f in files)}))
except Exception as e:
 (base/'verification-failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'command':command,'plan':plan},indent=2)+'\n');raise
