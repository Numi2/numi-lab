from pathlib import Path
import hashlib,json,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-bounded-drive-20261108'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
planpath=G/'plan.json';old=planpath.read_bytes();plan=json.loads(old)
p=G/'recover_git.py';new=G/'execute_git_recovery.py';assert not new.exists();p.rename(new)
p.write_bytes((R/'build/crow-bounded-drive-integration-build-01/recover_git.py').read_bytes())
plan['inputs']={str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [new,G/'cleanup_git_recovery.py']}
plan['recovery_entrypoint']=str(new.relative_to(R))
plan['revision_reason']='Archive verification correctly rejected a mutable raw-path backing after the instrument was changed. The original script bytes are restored; the revised instrument has a distinct filename. No native recovery or cleanup ran before repair.'
planpath.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
entries=tree();m=json.loads((R/plan['manifest']).read_bytes());mismatches=[]
with Objects() as objects:
 for f in m['files']:
  data=b''.join(content(f,entries,objects))
  if len(data)!=f['bytes'] or hashlib.sha256(data).hexdigest()!=f['sha256']:mismatches.append({'source':f['path'],'backing':f['backing']})
# HEAD still has the changed script until the next commit, so the mismatch
# remains visible here. The post-commit verifier must succeed before execution.
report={'failed_verification_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'failed_verification_error':'Recovered evidence mismatch: build/crow-bounded-drive-integration-build-01/recover_git.py','mismatches_before_repair_commit':mismatches,'previous_plan_sha256':hashlib.sha256(old).hexdigest(),'restored_backing_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'new_instrument_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'native_recovery_executed_before_repair':False,'source_deleted':False}
(G/'archive-verification-repair.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
