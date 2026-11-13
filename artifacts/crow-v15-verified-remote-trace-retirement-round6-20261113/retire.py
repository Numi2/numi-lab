from pathlib import Path
import hashlib,json,os,subprocess,sys
p=Path(sys.argv[1]);m=json.loads(p.read_text());root=Path(m['source_root']).resolve(strict=True);assert root==Path('/Users/n/Developer/numi-crow-rl-20261107');do_remove=len(sys.argv)>2 and sys.argv[2]=='--retire';rows=[]
for item in m['items']:
 rel=Path(item['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='artifacts' and rel.name=='run.stderr.gz';f=root/rel;assert not f.is_symlink() and f.resolve(strict=True)==f and f.is_file();st=f.stat();assert st.st_nlink==1 and st.st_size==item['bytes'];h=hashlib.sha256()
 with f.open('rb') as inp:
  for data in iter(lambda:inp.read(1048576),b''):h.update(data)
 assert h.hexdigest()==item['sha256'];l=subprocess.run(['lsof','-F','n','--',str(f)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout,(f,l.returncode,l.stdout,l.stderr)
 rows.append({'path':str(f),'sha256':h.hexdigest(),'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'verified_commit':item['verified_commit']})
if do_remove:
 for x in rows:
  f=Path(x['path']);st=f.stat();assert st.st_dev==x['device'] and st.st_ino==x['inode'] and st.st_nlink==1 and st.st_size==x['bytes'];l=subprocess.run(['lsof','-F','n','--',str(f)],capture_output=True,text=True);assert l.returncode==1 and not l.stdout;f.unlink()
print(json.dumps({'mode':'retired' if do_remove else 'checked','rows':rows,'bytes':sum(x['bytes'] for x in rows)},indent=2))
