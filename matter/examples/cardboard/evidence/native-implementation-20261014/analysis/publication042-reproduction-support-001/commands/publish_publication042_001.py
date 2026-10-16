from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication042-root-audit-001';e=r/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();git=lambda *v:subprocess.check_output(['git',*v],cwd=r,stderr=subprocess.STDOUT)
a=json.loads((d/'report.json').read_text());assert a['verdict']=='PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN';allowed={x.decode() for x in (d/'stage-paths.nul').read_bytes().split(b'\0') if x}
assert sha(e/'summary.json')==a['summary_sha256'] and sha(e/'publication-binding-reference042.json')==a['binding_sha256']
(d/'fetch.log').write_bytes(git('fetch','origin','main'));assert git('rev-parse','origin/main').decode().strip()==a['parent'];assert git('rev-parse','HEAD').decode().strip()==a['parent'] and git('branch','--show-current').decode().strip()=='main';assert not git('diff','--cached','--name-only')
git('add','-f','--pathspec-from-file='+str(d/'stage-paths.nul'),'--pathspec-file-nul');assert {x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}==allowed
msg=d/'commit-message.txt';msg.write_text("""Repair cardboard globalization registration and preserve source-bound evidence

Integrate the default-off best-of CGLS path, exact mortar-query observer and checked17-buffer contact plan. Repair five missing Runtime pipeline registrations after the preserved enabled608 crash. Ten targets build and14 CPU checks pass. Default-off611 reproduces32 production files; enabled612 completes all seven tiny stages under original physical acceptance criteria, selecting11 strictly lower CGLS trials with exact tested fractions.

Capture actual cached failing-state action602, retain its failed control19 history, and independently replay two exact mortar threshold crossings. Preserve trace field errata, failed builds and unavailable historical source reconstruction bytes. Whole-runtime shipping memory admission, conservative CCD repair, compact trajectory parity and the fresh full strict609 run remain open or active.

Publish only bounded solver/component progress. Full solver correctness, mesh/time convergence, material calibration, analogue validation, physical box assembly and crush/impact prediction remain unqualified. Preserve concurrent Human changes.
""")
(d/'commit.log').write_bytes(git('commit','-F',str(msg)));head=git('rev-parse','HEAD').decode().strip();assert {x.decode() for x in git('diff-tree','--no-commit-id','--name-only','-r','-z',head).split(b'\0') if x}==allowed
(d/'push.log').write_bytes(git('push','origin','main'));remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];assert remote==head;assert not git('status','--porcelain=v1','--untracked-files=all')
receipt={'schema':'numi.cardboard.publication042-commit-push.v1','parent':a['parent'],'commit':head,'origin_main':remote,'committed_paths':len(allowed),'root_audit_sha256':sha(d/'report.json'),'summary_sha256':sha(e/'summary.json'),'binding_sha256':sha(e/'publication-binding-reference042.json'),'source_frozen_sha256':a['source_frozen_sha256'],'all_six_full_sequence_axes_open':True,'canonical_tiny_native_trajectory_qualified':True,'full_target_trajectory_qualified':False,'active609_260_261_262_265_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'commit-push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
