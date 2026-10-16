from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication040-root-audit-002';e=r/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *v:subprocess.check_output(['git',*v],cwd=r,stderr=subprocess.STDOUT)
a=json.loads((d/'report.json').read_text());assert a['verdict']=='PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN'
allowed={x.decode() for x in (d/'stage-paths.nul').read_bytes().split(b'\0') if x}
assert sha(e/'summary.json')==a['summary_sha256'] and sha(e/'publication-binding-reference040.json')==a['binding_sha256']
(d/'fetch.log').write_bytes(git('fetch','origin','main'))
assert git('rev-parse','origin/main').decode().strip()==a['parent']
assert git('rev-parse','HEAD').decode().strip()==a['parent'] and git('branch','--show-current').decode().strip()=='main'
assert not git('diff','--cached','--name-only')
git('add','-f','--pathspec-from-file='+str(d/'stage-paths.nul'),'--pathspec-file-nul')
assert {x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}==allowed
msg=d/'commit-message.txt';msg.write_text("""Repair strict cardboard projection and admit actual refined coupon layouts

Add opt-in strict projected-state correction certification, checked32768-node pure FEM residual/CGLS arenas and exact27-refinement graph layouts. Honor shipping mesh/time controls. Compose bounded first-failure/contact-alpha diagnostics and correct large-node native measurement labels. Physical equations and acceptance tolerances remain unchanged.

The integrated build and eight CPU checks pass; a fresh canonical241 trajectory is pending. Native20682/32768 component checks pass without qualifying full FEM evolution. A complete strict64-Newton coupon still fails control19 despite satisfied true inner forcing.

Archive independent90-digit captured-state derivative/assembly evidence and additive corrections for glue omission and factor-two audit defects. Corrected contact traces retain exact production parity while exposing a legacy activation barrier rejection. Keep the separate damage-volume candidate and every failed/superseded attempt source-bound. Full convergence, calibration, assembly and crush/impact remain open.
""")
(d/'commit.log').write_bytes(git('commit','-F',str(msg)));head=git('rev-parse','HEAD').decode().strip()
assert {x.decode() for x in git('diff-tree','--no-commit-id','--name-only','-r','-z',head).split(b'\0') if x}==allowed
(d/'push.log').write_bytes(git('push','origin','main'))
remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];assert remote==head
assert not git('status','--porcelain=v1','--untracked-files=all')
receipt={'schema':'numi.cardboard.publication040-commit-push.v1','parent':a['parent'],'commit':head,'origin_main':remote,'committed_paths':len(allowed),'root_audit_sha256':sha(d/'report.json'),'summary_sha256':sha(e/'summary.json'),'binding_sha256':sha(e/'publication-binding-reference040.json'),'source_frozen_sha256':a['source_frozen_sha256'],'all_six_full_sequence_axes_open':True,'canonical_native_trajectory_pending':True,'active585_235_244_246_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'commit-push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
