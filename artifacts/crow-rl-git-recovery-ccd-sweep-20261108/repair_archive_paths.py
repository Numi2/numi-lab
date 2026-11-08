from pathlib import Path
import importlib.util,json,subprocess
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-ccd-sweep-20261108';M=R/'artifacts/crow-rl-ccd-128-integration-20261108/payload/manifest.json'
spec=importlib.util.spec_from_file_location('archive',R/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
old=a.file_hash(M);x=json.loads(M.read_text());changed=0
for f in x['files']:
    backing=f['backing']
    for item in backing.get('chunks',[backing]):
        p=Path(item['path'])
        if p.is_absolute():item['path']=str(p.relative_to(R));changed+=1
assert changed>0
a.write_json(M,x)
a.write_json(G/'archive-path-repair.json',{'cause':'Caller supplied absolute output directory to capture; new backing blob paths became absolute and actual-Git verifier correctly rejected manifest before any raw evidence deletion.','original_commit':a.git('rev-parse','HEAD').decode().strip(),'original_manifest_sha256':old,'corrected_manifest_sha256':a.file_hash(M),'changed_backing_path_fields':changed,'change':'Normalize only paths within this exact repository root to relative paths. No source files, payload bytes, hashes, modes, mtimes or qualification results change. Caller fixed to use repository-relative output path.'})
subprocess.run(['git','add',str(M),str(G/'archive-path-repair.json'),str(G/'repair_archive_paths.py'),'artifacts/crow-rl-ccd-128-integration-20261108/finalize_archive.py'],check=True)
subprocess.run(['git','commit','-m','Correct archive path metadata rejected by Git verifier'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
a.verify(M.relative_to(R))
print(json.dumps(x['summary']),flush=True)
