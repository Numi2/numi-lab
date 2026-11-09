from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd(); root=R/'build/crow-phase-restore-20261109';manifest=R/'artifacts/crow-flight-envelope-recovery-20261109/manifest.json'
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0]
if head!=remote:raise RuntimeError('Push current HEAD before retirement')
subprocess.run(['python3','tools/crow_evidence_archive.py','verify',str(manifest.relative_to(R)),'--prefix','build/crow-flight-envelope-20261109'],check=True)
m=json.loads(manifest.read_text());entries=[x for x in m['files'] if x['path'].startswith('build/crow-flight-envelope-20261109/')]
expected={root/x['path']:x for x in entries}
actual=set()
for directory,dirs,names in os.walk(root,followlinks=False):
 for name in dirs:
  if (Path(directory)/name).is_symlink():raise RuntimeError('Unexpected symlink')
 for name in names:
  p=Path(directory)/name
  if p.is_symlink() or not p.is_file():raise RuntimeError('Unexpected file type')
  actual.add(p)
if actual!=set(expected):raise RuntimeError('Restored copy differs in file inventory')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
for p,f in expected.items():
 if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise RuntimeError('Changed copy: '+str(p))
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(root)],capture_output=True,text=True,timeout=60)
if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open handles or lsof error')
before=subprocess.check_output(['df','-k','.'],text=True)
for p in expected:p.unlink()
for directory,_,_ in os.walk(root,topdown=False):Path(directory).rmdir()
after=subprocess.check_output(['df','-k','.'],text=True)
receipt={'head':head,'remote':remote,'source_manifest':str(manifest.relative_to(R)),'source_manifest_sha256':sha(manifest),'removed_copy_root':str(root.relative_to(R)),'files_verified_and_removed':len(expected),'logical_bytes':sum(x['bytes'] for x in entries),'open_handles':0,'df_before':before,'df_after':after}
Path('artifacts/crow-restored-copy-cleanup-20261109/receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
