from pathlib import Path
import json,gzip,hashlib,shutil,subprocess,os,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
assert len(sys.argv)==2 and '/' not in sys.argv[1]
a,b=D/'shared-three-rhs-native-190',D/sys.argv[1]
assert (b/'install-motion.execution.json').exists()
assert subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).returncode!=0
def sha(p,op=open):
 h=hashlib.sha256()
 with op(p,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
r={'before_available_bytes':shutil.disk_usage(D).free,'compression':{},'clones':[]}
src=b/'install-motion.stdout.log';dst=src.with_suffix(src.suffix+'.gz')
if src.exists():
 h=sha(src)
 with src.open('rb') as fi,gzip.open(dst,'wb',compresslevel=1) as fo:shutil.copyfileobj(fi,fo,4*1024*1024)
 assert sha(dst,gzip.open)==h
 r['compression']={'original_bytes':src.stat().st_size,'gzip_bytes':dst.stat().st_size,'original_and_restored_sha256':h}
 src.unlink()
for p in sorted((a/'install-motion').rglob('*')):
 if not p.is_file() or p.is_symlink() or p.stat().st_size<3*1024**2:continue
 rel=p.relative_to(a/'install-motion');target=b/'install-motion'/rel
 if not target.is_file() or target.is_symlink() or target.stat().st_size!=p.stat().st_size:continue
 hx=sha(p)
 if sha(target)!=hx:continue
 tmp=target.with_name(target.name+'.verified-clone')
 subprocess.run(['/bin/cp','-c','-p',str(p),str(tmp)],check=True)
 assert sha(tmp)==hx and tmp.stat().st_ino!=p.stat().st_ino
 os.replace(tmp,target)
 r['clones'].append({'path':str(rel),'canonical':str(p),'sha256':hx,'bytes':p.stat().st_size,'independent_inode':True})
r['after_available_bytes']=shutil.disk_usage(D).free
(b/'verified-compression-clones.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='clones'},indent=2))
print('clones',len(r['clones']))

