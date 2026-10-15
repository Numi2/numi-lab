#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
BASE=B/'snapshots/source-cgls-post-release-relaxation-145'
TARGET=B/'snapshots/source-cgls-block-right-preconditioned-149'
OVER=B/'analysis/native-cgls-block-right-preconditioning-overlay-149-002'
OUT=B/'analysis/source149-right-block-cgls-integration-001'
BUILD=R/'build-cardboard-provenance-149'
MIN_FREE=1536*1024**2
CHANGED=['matter/src/runtime.mm','matter/src/metal/fgmres.metalinc','matter/src/metal/base_cgls.metalinc']

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def free_bytes(path=B):
 st=os.statvfs(path); return st.f_bavail*st.f_frsize
def inventory(root:Path):
 result={}
 for current,dirs,files in os.walk(root,followlinks=False):
  dirs.sort(); files.sort(); cp=Path(current)
  for name in files:
   p=cp/name; rel=str(p.relative_to(root))
   if p.is_symlink(): result[rel]={'kind':'symlink','target':os.readlink(p)}
   elif p.is_file(): result[rel]={'kind':'file','bytes':p.stat().st_size,'sha256':sha(p)}
 return result

def main():
 if TARGET.exists() or BUILD.exists() or (OUT.exists() and any(q.name!='assemble_source149.py' for q in OUT.iterdir())):
  raise SystemExit('target snapshot/build/output already exists; refusing overwrite')
 before=free_bytes()
 if before < MIN_FREE: raise SystemExit(f'free space below 1.5 GiB guard: {before}')
 basef=json.loads((BASE/'frozen-source.json').read_text())
 bases=json.loads((BASE/'configure-support-source.json').read_text())
 overlay=json.loads((OVER/'source-overlay.json').read_text())
 assert overlay['overlay_id']=='native-cgls-block-right-preconditioning-149-002'
 assert overlay['base_snapshot']==str(BASE)
 assert len(basef['source_sha256'])==787 and len(bases['files'])==1034
 for rel,digest in basef['source_sha256'].items():
  p=BASE/rel
  if not p.is_file() or sha(p)!=digest: raise SystemExit(f'base source hash mismatch: {rel}')
 for rel,digest in bases['files'].items():
  p=BASE/rel
  if not p.is_file() or sha(p)!=digest: raise SystemExit(f'configure support hash mismatch: {rel}')
 for rel in CHANGED:
  if sha(BASE/rel)!=overlay['base_source_hashes'][rel]: raise SystemExit(f'overlay base mismatch: {rel}')
  if sha(OVER/'source'/rel)!=overlay['overlay_source_hashes'][rel]: raise SystemExit(f'overlay result mismatch: {rel}')
 if sha(OVER/'source-overlay.patch')!=overlay['patch_sha256']: raise SystemExit('overlay patch hash mismatch')
 OUT.mkdir(exist_ok=True)
 before_inv=inventory(BASE)
 (OUT/'preflight.json').write_text(json.dumps({'base_snapshot':str(BASE),'base_frozen_sha256':sha(BASE/'frozen-source.json'),'base_source_count':len(basef['source_sha256']),'base_configure_count':len(bases['files']),'base_inventory_count':len(before_inv),'overlay_manifest_sha256':sha(OVER/'source-overlay.json'),'overlay_patch_sha256':sha(OVER/'source-overlay.patch'),'changed_paths':CHANGED,'available_bytes_before_clone':before,'minimum_free_bytes':MIN_FREE,'no_gpu_or_probe_execution':True},indent=2,sort_keys=True)+'\n')
 cp=subprocess.run(['/bin/cp','-cR',str(BASE),str(TARGET)],capture_output=True,text=True)
 (OUT/'clone.stdout.log').write_text(cp.stdout); (OUT/'clone.stderr.log').write_text(cp.stderr)
 if cp.returncode: raise SystemExit(f'APFS clone failed {cp.returncode}: {cp.stderr}')
 patch=subprocess.run(['patch','--batch','--forward','-p1','-d',str(TARGET),'-i',str(OVER/'source-overlay.patch')],capture_output=True,text=True)
 (OUT/'patch.stdout.log').write_text(patch.stdout); (OUT/'patch.stderr.log').write_text(patch.stderr)
 if patch.returncode: raise SystemExit(f'patch failed {patch.returncode}: {patch.stderr}')
 frozen=json.loads((TARGET/'frozen-source.json').read_text())
 assert frozen['source_sha256']==basef['source_sha256']
 for rel in CHANGED: frozen['source_sha256'][rel]=sha(TARGET/rel)
 frozen['base_snapshot']=str(BASE)
 frozen['scope']=('Source145 plus a default-off, bounded block3 right-preconditioned truncated-CGLS direction experiment, '
  'selected only through the pre-existing CGLS fallback scope. The physical operator, trial Armijo residual, final '
  'direction installer, original Newton certificate checks, material law, and acceptance thresholds are unchanged. '
  'The new block builder writes to private scratch. This is an unqualified conditioning experiment; no solver or physical '
  'qualification is implied.')
 (TARGET/'frozen-source.json').write_text(json.dumps(frozen,indent=2)+'\n')
 if (TARGET/'configure-support-source.json').read_bytes()!=(BASE/'configure-support-source.json').read_bytes(): raise SystemExit('configure-support manifest changed')
 for rel,digest in frozen['source_sha256'].items():
  if sha(TARGET/rel)!=digest: raise SystemExit(f'target frozen source hash mismatch {rel}')
 for rel,digest in bases['files'].items():
  if sha(TARGET/rel)!=digest: raise SystemExit(f'target configure hash mismatch {rel}')
 after_inv=inventory(TARGET)
 if before_inv.keys()!=after_inv.keys(): raise SystemExit('file path inventory changed')
 allowed=set(CHANGED)|{'frozen-source.json'}
 delta={rel for rel in before_inv if before_inv[rel]!=after_inv[rel]}
 if delta!=allowed: raise SystemExit(f'unexpected snapshot delta {sorted(delta^allowed)}')
 command=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(TARGET/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(TARGET),'--build-dir',str(BUILD),'--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe']
 (OUT/'build-command.json').write_text(json.dumps(command,indent=2)+'\n')
 integration={'schema':'numi.cardboard.source149_right_block_cgls_integration.v1','snapshot':str(TARGET),'base_snapshot':str(BASE),'base_frozen_sha256':sha(BASE/'frozen-source.json'),'frozen_source_sha256':sha(TARGET/'frozen-source.json'),'source_hash_entry_count':len(frozen['source_sha256']),'configure_support_sha256':sha(TARGET/'configure-support-source.json'),'configure_support_entry_count':len(bases['files']),'configure_support_hashes_preserved':True,'overlay_manifest_sha256':sha(OVER/'source-overlay.json'),'overlay_patch_sha256':sha(OVER/'source-overlay.patch'),'changed_files':{rel:{'bytes':(TARGET/rel).stat().st_size,'sha256':sha(TARGET/rel)} for rel in CHANGED+['frozen-source.json']},'all_other_source145_files_byte_identical':True,'full_inventory_count':len(after_inv),'unchanged_inventory_count':len(after_inv)-len(allowed),'disk_available_bytes_before_clone':before,'disk_available_bytes_after_clone':free_bytes(),'build_directory':str(BUILD),'build_command':command,'build_status':'prepared','native_probe_or_gpu_launch':'not run'}
 (OUT/'integration.json').write_text(json.dumps(integration,indent=2,sort_keys=True)+'\n')
 (OUT/'source149-inventory.json').write_text(json.dumps(after_inv,indent=2,sort_keys=True)+'\n')
 (OUT/'source145-inventory.json').write_text(json.dumps(before_inv,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'snapshot':str(TARGET),'integration':str(OUT/'integration.json'),'build_command':command,'disk_before':before,'disk_after_clone':free_bytes(),'files':len(after_inv),'changed':sorted(delta)},indent=2))
if __name__=='__main__': main()
