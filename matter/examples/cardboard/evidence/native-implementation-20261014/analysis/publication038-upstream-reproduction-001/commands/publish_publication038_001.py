from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication038-root-audit-001';e=r/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *v:subprocess.check_output(['git',*v],cwd=r,stderr=subprocess.STDOUT)
a=json.loads((d/'report.json').read_text());assert a['verdict']=='PASS_WITH_INVALID_541_FD_EVIDENCE_WITHDRAWN'
allowed={x.decode() for x in (d/'stage-paths.nul').read_bytes().split(b'\0') if x}
assert sha(e/'summary.json')==a['summary_sha256'] and sha(e/'publication-binding-reference038.json')==a['binding_sha256']
(d/'fetch.log').write_bytes(git('fetch','origin','main'))
assert git('rev-parse','origin/main').decode().strip()==a['parent']
assert git('rev-parse','HEAD').decode().strip()==a['parent'] and git('branch','--show-current').decode().strip()=='main'
assert not git('diff','--cached','--name-only')
git('add','-f','--pathspec-from-file='+str(d/'stage-paths.nul'),'--pathspec-file-nul')
assert {x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}==allowed
msg=d/'commit-message.txt';msg.write_text("""Gate cardboard capacity on damage and repair checkpoint replay proof

Require compatible irreversible quadrature-area debond comparisons and keep enabled but unmeasured paper damage from qualifying Fpeak. Permit diagnostic replay from an output-free source-bound qualification runner, retaining exact contracts and the corrected53-key registry.

Fresh source222 build,31 capacity checks,12 proof-admission cases and the existing checkpoint checker pass. Preserve native545 control19 failure and coupled contact activation failures. Withdraw invalid541 finite-difference comparisons caused by aliased perturbation inputs and retain corrected547 mixed results.

Archive65 closed evidence cases with source and payload hashes. Full solver, convergence, calibration, assembly and crush/impact qualification remain open; active555 is excluded.
""")
(d/'commit.log').write_bytes(git('commit','-F',str(msg)));head=git('rev-parse','HEAD').decode().strip()
assert {x.decode() for x in git('diff-tree','--no-commit-id','--name-only','-r','-z',head).split(b'\0') if x}==allowed
(d/'push.log').write_bytes(git('push','origin','main'))
remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];assert remote==head
assert not git('status','--porcelain=v1','--untracked-files=all')
receipt={'schema':'numi.cardboard.publication038-commit-push.v1','parent':a['parent'],'commit':head,'origin_main':remote,'committed_paths':len(allowed),'root_audit_sha256':sha(d/'report.json'),'summary_sha256':sha(e/'summary.json'),'binding_sha256':sha(e/'publication-binding-reference038.json'),'all_six_full_sequence_axes_open':True,'active555_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'commit-push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
