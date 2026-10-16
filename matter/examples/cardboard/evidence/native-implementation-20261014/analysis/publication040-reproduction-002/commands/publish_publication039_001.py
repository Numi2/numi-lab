from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication039-root-audit-002';e=r/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *v:subprocess.check_output(['git',*v],cwd=r,stderr=subprocess.STDOUT)
a=json.loads((d/'report.json').read_text());assert a['verdict']=='PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN'
allowed={x.decode() for x in (d/'stage-paths.nul').read_bytes().split(b'\0') if x}
assert sha(e/'summary.json')==a['summary_sha256'] and sha(e/'publication-binding-reference039.json')==a['binding_sha256']
(d/'fetch.log').write_bytes(git('fetch','origin','main'))
assert git('rev-parse','origin/main').decode().strip()==a['parent']
assert git('rev-parse','HEAD').decode().strip()==a['parent'] and git('branch','--show-current').decode().strip()=='main'
assert not git('diff','--cached','--name-only')
git('add','-f','--pathspec-from-file='+str(d/'stage-paths.nul'),'--pathspec-file-nul')
assert {x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}==allowed
msg=d/'commit-message.txt';msg.write_text("""Expand bounded cardboard solver arenas and preserve exact diagnostic replay

Add checked pure-FEM true-residual and CGLS workspace plans for8192 nodes, plus bounded free-box proxy admission, without changing the physical operator. Admit proof-only finite-difference epsilon controls for exact accepted-checkpoint replay.

Manufactured native component checks pass. Two fresh composed builds complete all seven phases of the tiny coupon with32 production files byte-identical to the prior source210 regression. Native555 and565 retain the original19th-control failure and exact accepted/rejected production suffixes. Fine-perturbation derivatives remain inconclusive; the targeted strict local-projection candidate and cohesive page-kernel results are archived separately and do not qualify the full runtime.

Archive60 closed cases and preserve2002 prior records with exact source and payload hashes. Fresh source234 build and nine CPU checks pass after disjoint Human upstream changes. Full solver correctness, mesh/time convergence, calibration, physical box assembly and crush/impact qualification remain open.
""")
(d/'commit.log').write_bytes(git('commit','-F',str(msg)));head=git('rev-parse','HEAD').decode().strip()
assert {x.decode() for x in git('diff-tree','--no-commit-id','--name-only','-r','-z',head).split(b'\0') if x}==allowed
(d/'push.log').write_bytes(git('push','origin','main'))
remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];assert remote==head
assert not git('status','--porcelain=v1','--untracked-files=all')
receipt={'schema':'numi.cardboard.publication039-commit-push.v1','parent':a['parent'],'commit':head,'origin_main':remote,'committed_paths':len(allowed),'root_audit_sha256':sha(d/'report.json'),'summary_sha256':sha(e/'summary.json'),'binding_sha256':sha(e/'publication-binding-reference039.json'),'source_frozen_sha256':a['source_frozen_sha256'],'all_six_full_sequence_axes_open':True,'active232_233_235_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'commit-push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
