from pathlib import Path
import json,hashlib,gzip,os,subprocess,re,datetime
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'matter/examples/cardboard/evidence/native-implementation-20261014';O=B/'analysis/closed-published-output-compaction-021';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();av=lambda:os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
summary=json.loads((D/'summary.json').read_text());before=av();selected=[];skipped=[]
for rec in summary['records']:
 c=rec.get('case','');ns=re.findall(r'(?<!\d)(\d{3})(?!\d)',c.rsplit('/',1)[-1])
 if not c.startswith(('solver/','native/')) or not ns or int(ns[-1])>414:continue
 root=B/c;receipt=root/'receipt.json'
 rows=[x for x in rec.get('files',[]) if x.get('bytes',0)>=1024**2 and x.get('repository_compressed_copy') and Path(x.get('external_path','/nonexistent')).is_file()]
 if not rows:continue
 if not receipt.is_file():skipped.append({'case':c,'reason':'no terminal receipt'});continue
 rr=json.loads(receipt.read_text())
 if rr.get('exit_code') is None or rr.get('timed_out'):skipped.append({'case':c,'reason':'receipt incomplete or timeout; retained'});continue
 ls=subprocess.run(['lsof','-nP','+D',str(root)],capture_output=True,text=True)
 if ls.stdout.strip() or ls.stderr.strip() or ls.returncode not in (0,1):skipped.append({'case':c,'reason':'open handle or lsof error; retained'});continue
 for x in rows:
  p=Path(x['external_path']);p.relative_to(root);assert not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];archive=D/x['repository_compressed_copy'];assert sha(archive)==x['compressed_sha256'];rel=str(archive.relative_to(R));headblob=subprocess.check_output(['git','rev-parse','HEAD:'+rel],cwd=R,text=True).strip();diskblob=subprocess.check_output(['git','hash-object',str(archive)],cwd=R,text=True).strip();assert headblob==diskblob
  tmp=O/'restore-check.bin';h=hashlib.sha256();n=0
  with gzip.open(archive,'rb') as src,tmp.open('xb') as out:
   for block in iter(lambda:src.read(1024**2),b''):out.write(block);h.update(block);n+=len(block)
   out.flush();os.fsync(out.fileno())
  assert n==x['bytes'] and h.hexdigest()==x['sha256'] and sha(tmp)==x['sha256'];tmp.unlink()
  selected.append({'case':c,'original':str(p),'sha256':x['sha256'],'bytes':x['bytes'],'allocated_bytes':p.stat().st_blocks*512,'archive':str(archive),'archive_sha256':x['compressed_sha256'],'archive_git_blob':headblob,'receipt_sha256':sha(receipt),'restoration_verified':True})
(O/'before-manifest.json').write_text(json.dumps({'summary_sha256':sha(D/'summary.json'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'before_available_bytes':before,'selected':selected,'skipped':skipped},indent=2)+'\n')
for c in sorted({x['case'] for x in selected}):
 root=B/c;ls=subprocess.run(['lsof','-nP','+D',str(root)],capture_output=True,text=True);assert not ls.stdout.strip() and not ls.stderr.strip() and ls.returncode in (0,1)
 rows=[x for x in selected if x['case']==c];pointer=root/'outputs.compaction021.json';assert not pointer.exists();pointer.write_text(json.dumps({'schema':'numi.cardboard.published_output_duplicate_compaction.v1','files':rows,'restore_script':str(O/'restore.py'),'reason':'External duplicate removed only after committed repository archive hash, full decompression restore and original-file revalidation; all source inputs/binaries/receipts retained.'},indent=2)+'\n')
 for x in rows:
  p=Path(x['original']);assert p.stat().st_nlink==1 and sha(p)==x['sha256'] and sha(Path(x['archive']))==x['archive_sha256'];p.unlink()
 print(c,len(rows),flush=True)
(O/'restore.py').write_text('''from pathlib import Path\nimport gzip,hashlib,json,sys,os\np=json.loads(Path(sys.argv[1]).read_text())\nfor r in p['files']:\n a=Path(r['archive']);o=Path(r['original']);assert hashlib.sha256(a.read_bytes()).hexdigest()==r['archive_sha256'];assert not o.exists();o.parent.mkdir(parents=True,exist_ok=True)\n with gzip.open(a,'rb') as src,o.open('xb') as out:\n  for b in iter(lambda:src.read(1024**2),b''):out.write(b)\n  out.flush();os.fsync(out.fileno())\n assert o.stat().st_size==r['bytes'] and hashlib.sha256(o.read_bytes()).hexdigest()==r['sha256']\nprint('All archived original bytes restored and verified')\n''')
report={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_duplicate_files_removed':len(selected),'logical_bytes':sum(x['bytes'] for x in selected),'allocated_bytes':sum(x['allocated_bytes'] for x in selected),'before_available_bytes':before,'after_available_bytes':av(),'raw_source_inputs_removed':0,'source_snapshots_or_native_binaries_removed':0,'all_repository_archives_match_committed_HEAD_and_restore':True};(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'runner.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.iterdir()) if p.is_file()}},indent=2)+'\n');print(json.dumps(report),flush=True)
