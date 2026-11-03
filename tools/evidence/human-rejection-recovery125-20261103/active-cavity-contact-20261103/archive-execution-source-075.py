from pathlib import Path
import subprocess,hashlib,json,tarfile,tempfile,shutil,gzip,io
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
A=D/'completed-incidence-067'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',head],cwd=R,text=True).splitlines())
changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=R,text=True).splitlines()
patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=R)
extensions={'.c','.cc','.cpp','.h','.hh','.hpp','.m','.mm','.metal','.metalinc','.py','.cmake','.inc','.in','.swift','.def','.plist'}
is_source=lambda f:'evidence' not in Path(f).parts and (Path(f).suffix in extensions or Path(f).name=='CMakeLists.txt')
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=R,text=True).splitlines()
untracked=[f for f in untracked if is_source(f) and (R/f).is_file() and not (R/f).is_symlink()]
inputs=set()
for target in ['bin/numi-human-native','matter/numi-matter-fgmres-true-residual-native-check','matter/numi-matter-vascular-cavity-check']:
 text=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build-native'),'-t','inputs',target],text=True)
 for raw in text.splitlines():
  p=Path(raw)
  if not p.is_absolute():p=R/'build-native'/p
  p=p.resolve()
  if p.is_file() and p.is_relative_to(R) and not p.is_relative_to(R/'build-native'):inputs.add(str(p.relative_to(R)))
basepaths=sorted({f for f in tracked if is_source(f)}|(tracked&set(changed))|(tracked&inputs))
base=subprocess.check_output(['git','archive',head,*basepaths],cwd=R)
extra=sorted(set(untracked)|(inputs-tracked))
manifest={'base_revision':head,'base_code_paths':basepaths,'base_code_tar_sha256':hashlib.sha256(base).hexdigest(),
 'dirty_patch_sha256':hashlib.sha256(patch).hexdigest(),
 'tracked_changed':{f:sha(R/f) if (R/f).is_file() else None for f in changed},
 'source_overlays':{f:sha(R/f) for f in extra},
 'actual_build_source_inputs':{f:sha(R/f) for f in sorted(inputs)},
 'scope':'source code recovery; base code tar, tracked dirty patch and untracked/build-source overlays. Original anatomical assets and SDK remain separately pinned; historical evidence excluded.'}
out=A/'execution-source-recovery.tar.gz'
with tarfile.open(out,'w:gz') as tar:
 for name,data in [('_execution_recovery/base-code.tar',base),('_execution_recovery/execution-source.patch',patch),('_execution_recovery/manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())]:
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 for f in extra:tar.add(R/f,arcname=f)
assert out.stat().st_size<100*2**20
with tempfile.TemporaryDirectory(prefix='numi-source-restore-',dir='/tmp') as raw:
 temp=Path(raw)
 with tarfile.open(fileobj=io.BytesIO(base)) as tar:tar.extractall(temp,filter='data')
 patchpath=temp/'execution-source.patch';patchpath.write_bytes(patch)
 subprocess.run(['git','apply','--check',str(patchpath)],cwd=temp,check=True,capture_output=True)
 subprocess.run(['git','apply',str(patchpath)],cwd=temp,check=True,capture_output=True)
 with tarfile.open(out,'r:gz') as tar:tar.extractall(temp,filter='data')
 for f,digest in manifest['tracked_changed'].items():assert (sha(temp/f) if (temp/f).is_file() else None)==digest,f
 for key in ['source_overlays','actual_build_source_inputs']:
  for f,digest in manifest[key].items():assert sha(temp/f)==digest,f
receipt={'archive_path':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'base_revision':head,
 'base_code_files':len(basepaths),'tracked_dirty_files_restored':len(changed),'source_overlays_restored':len(extra),
 'actual_build_source_inputs_verified':len(inputs),'restore_test':'all captured source SHA256 equal after base-code extraction, patch application and overlays',
 'restore_order':['extract _execution_recovery/base-code.tar into a fresh source directory','git apply _execution_recovery/execution-source.patch from that directory','extract source overlays from the outer archive','verify _execution_recovery/manifest.json SHA256 entries'],
 'runtime_assets':'preserved and pinned separately; not uploaded or relicensed'}
(A/'source-recovery-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('source_restore_verified',json.dumps(receipt),flush=True)
p=D/'final-combined-capture-068-retry1/install-motion.stdout.log'
original=sha(p);assert original==json.loads((p.parent/'install-motion.execution.json').read_text())['stdout_sha256']
stat=p.stat();out=p.with_suffix(p.suffix+'.gz')
with p.open('rb') as src,gzip.open(out,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst,2**20)
with gzip.open(out,'rb') as src:assert hashlib.file_digest(src,'sha256').hexdigest()==original
assert p.stat().st_mtime_ns==stat.st_mtime_ns
receipt={'original_path':str(p),'original_bytes':stat.st_size,'original_sha256':original,'gzip_path':str(out),'gzip_bytes':out.stat().st_size,'gzip_sha256':sha(out),'restore_command':"gzip -dc '"+str(out)+"' > '"+str(p)+"'",'recovered_bytes':stat.st_size-out.stat().st_size}
p.unlink()
cleanup=D/'lossless-log-cleanup-075.json';m=json.loads(cleanup.read_text());m['files'].append(receipt);cleanup.write_text(json.dumps(m,indent=2)+'\n')
resources=[]
for row in json.loads((D/'paired-performance-068-074.json').read_text()):
 p=D/row['name'];samples=[json.loads(line) for line in (p/'resource-monitor.jsonl').read_text().splitlines()]
 baseline=json.loads((p/'resource-policy.json').read_text())['baseline_swap_used_mib']
 resources.append({'name':row['name'],'minimum_disk_GiB':min(s['disk_available_bytes'] for s in samples)/2**30,'maximum_additional_swap_MiB':max(s['swap_used_mib'] for s in samples)-baseline})
(D/'resource-summary-068-074.json').write_text(json.dumps({'successful_runs':resources,'available_disk_after_cleanup_bytes':shutil.disk_usage(D).free},indent=2)+'\n')
print('free_GiB',shutil.disk_usage(D).free/2**30,flush=True)
