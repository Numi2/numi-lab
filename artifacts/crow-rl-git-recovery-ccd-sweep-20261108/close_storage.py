from pathlib import Path
import gzip,importlib.util,json,stat,subprocess
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-ccd-sweep-20261108';A=R/'artifacts/crow-rl-ccd-128-integration-20261108'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
M=A.relative_to(R)/'payload/manifest.json'
assert json.loads((G/'executed-recovery.json').read_text())['status']=='verified'
assert not (R/'build/crow-ccd-sweep-git-recovery-01').exists()
def commit(message,paths):
    subprocess.run(['git','add',*[str(p) for p in paths]],check=True)
    subprocess.run(['git','commit','-m',message],check=True)
    subprocess.run(['git','push','origin','HEAD'],check=True)
# Preserve exact original private wrapper logs before removing duplicates.
prefixes=('crow-ccd-event-capture-','crow-ccd-contact-time-','crow-ccd-articulated-sweep-','crow-ccd-sweep-')
logs=sorted(p for p in Path('/private/tmp').iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix in ('.stdout','.stderr'))
opened=subprocess.run(['/usr/sbin/lsof','-Fpn',*[str(p) for p in logs]],capture_output=True,text=True,timeout=60)
assert opened.returncode in (0,1) and not opened.stdout and not opened.stderr
D=G/'wrapper-logs';D.mkdir(exist_ok=False);inventory=[]
for p in logs:
    s=p.stat();dest=D/(p.name+'.gz');dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    inventory.append({'path':str(p),'archive':str(dest.relative_to(R)),'sha256':a.file_hash(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns})
a.write_json(G/'wrapper-inventory.json',{'files':inventory,'open_handles':0})
commit('Verify executed Git recovery and retain all crow study wrapper logs',[G])
a.prune(M,A.relative_to(R)/'cleanup-receipt.json')
commit('Prune only Git-verified and replayed crow evidence copies',[A/'cleanup-receipt.json'])
entries=a.tree();before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
opened=subprocess.run(['/usr/sbin/lsof','-Fpn',*[f['path'] for f in inventory]],capture_output=True,text=True,timeout=60)
assert opened.returncode in (0,1) and not opened.stdout and not opened.stderr
with a.Objects() as obj:
    for f in inventory:
        raw=gzip.decompress(obj.read(entries[f['archive']][1]));p=Path(f['path']);s=p.stat()
        assert raw==p.read_bytes() and (a.file_hash(p),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
for f in inventory:Path(f['path']).unlink()
a.write_json(G/'wrapper-cleanup.json',{'files':inventory,'actual_git_bytes_verified':True,'open_handles':0,'removed_bytes':sum(f['bytes'] for f in inventory),'df_before':before,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)})
commit('Remove archived closed wrapper log duplicates',[G/'wrapper-cleanup.json'])
a.omit_archives(G.relative_to(R)/'omission.json')
commit('Omit byte-verified evidence archive checkout duplicates',[G/'omission.json'])
subprocess.run(['git','diff','--check'],check=True)
assert not a.git('status','--porcelain')
head=a.git('rev-parse','HEAD').decode().strip();remote=a.git('ls-remote','origin','refs/heads/'+a.BRANCH).decode().split()[0];assert head==remote
policy=R/'environments/american-crow/action-holds-v1/policies/deployment97.policypack'
assert a.file_hash(policy)=='6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389'
workers=subprocess.check_output(['ps','-axo','pid,ppid,%cpu,command'],text=True)
owned=[line for line in workers.splitlines() if any(str(R/'build'/p) in line for p in ('crow-ccd-event-capture-build-01/','crow-ccd-contact-time-build-01/','crow-ccd-articulated-sweep-build-01/','crow-ccd-sweep-git-recovery-01/'))]
assert not owned,owned
boundary={'verified_head_before_receipt':head,'remote_head_before_receipt':remote,'worktree_clean_before_receipt':True,'diff_check_passed':True,'owned_native_workers':owned,'protected_policy_sha256':a.file_hash(policy),'df_data':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True),'qualification':'Full mechanical release convergence still fails62.607mm versus5mm at64/128. No policy/profile promotion. Broad lifelong environment goal remains unfinished.','scope':'Storage closure for this set of four studies; all494 raw files and original failures archived in actual Git bytes, nine scoped restored native executions verified.'}
a.write_json(G/'storage-and-scope-boundary.json',boundary)
commit('Record clean crow evidence recovery and unchanged deployment boundary',[G/'storage-and-scope-boundary.json'])
assert not a.git('status','--porcelain')
head=a.git('rev-parse','HEAD').decode().strip();remote=a.git('ls-remote','origin','refs/heads/'+a.BRANCH).decode().split()[0];assert head==remote
print(json.dumps({'status':'storage_closed','head':head,'remote_head':remote,'raw_removed_bytes':json.loads((A/'cleanup-receipt.json').read_text())['removed_logical_bytes'],'temp_recovery_removed_bytes':json.loads((G/'executed-recovery.json').read_text())['temporary_cleanup']['removed_bytes'],'archive_checkout_omitted_bytes':json.loads((G/'omission.json').read_text())['omitted_logical_bytes'],'private_wrapper_removed_bytes':sum(f['bytes'] for f in inventory),'df_data':boundary['df_data'],'policy_unchanged':True,'workers':owned},indent=2),flush=True)
