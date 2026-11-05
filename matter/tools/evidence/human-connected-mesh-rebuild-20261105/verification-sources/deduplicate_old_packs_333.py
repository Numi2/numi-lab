from pathlib import Path
import os,json,hashlib,subprocess,shutil,time,stat
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'resource-recovery-333'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=[]
for directory in sorted(W.iterdir()):
 if not directory.is_dir() or directory.is_symlink():continue
 try:trial=int(directory.name.rsplit('-',1)[1])
 except (ValueError,IndexError):continue
 if trial>=307:continue
 for p in directory.rglob('*.mrvpack'):
  if p.is_symlink():continue
  s=p.stat()
  if s.st_size<100*1024**2:continue
  files.append(p)
groups={}
for p in files:
 s=p.stat();key=(s.st_size,sha(p),stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid)
 groups.setdefault(key,[]).append(p)
rows=[];before=shutil.disk_usage(W).free
for key,paths in groups.items():
 if len(paths)<2:continue
 canonical=paths[0]
 assert sha(canonical)==key[1]
 for p in paths[1:]:
  s=p.stat();cs=canonical.stat()
  if s.st_ino==cs.st_ino:continue
  assert s.st_dev==cs.st_dev
  assert subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True).returncode==1
  assert subprocess.run(['/usr/sbin/lsof','-t','--',str(canonical)],capture_output=True).returncode==1
  assert sha(p)==key[1]
  row={'path':str(p),'canonical':str(canonical),'sha256':key[1],'bytes':s.st_size,'old_inode':s.st_ino,'old_mtime_ns':s.st_mtime_ns}
  temp=p.with_name(p.name+'.verified-link333');assert not temp.exists()
  os.link(canonical,temp);os.replace(temp,p)
  assert p.stat().st_ino==canonical.stat().st_ino
  rows.append(row)
os.sync()
receipt={'status':'PASS_EXACT_BYTE_DUPLICATES_SHARED','files_scanned':len(files),'distinct_sha_groups':len(groups),'paths_deduplicated':len(rows),'unchanged_bytes_and_paths':True,'all_targets_and_canonical_files_had_no_open_handles':True,'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(W).free,'operations':rows}
(D/'duplicate-packs-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='operations'},indent=2))
