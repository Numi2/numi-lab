from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';d=q/'analysis/publication041-root-audit-001';e=r/'matter/examples/cardboard/evidence/native-implementation-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();git=lambda *v:subprocess.check_output(['git',*v],cwd=r,stderr=subprocess.STDOUT)
a=json.loads((d/'report.json').read_text());assert a['verdict']=='PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN';allowed={x.decode() for x in (d/'stage-paths.nul').read_bytes().split(b'\0') if x}
assert sha(e/'summary.json')==a['summary_sha256'] and sha(e/'publication-binding-reference041.json')==a['binding_sha256']
(d/'fetch.log').write_bytes(git('fetch','origin','main'));assert git('rev-parse','origin/main').decode().strip()==a['parent'];assert git('rev-parse','HEAD').decode().strip()==a['parent'] and git('branch','--show-current').decode().strip()=='main';assert not git('diff','--cached','--name-only')
git('add','-f','--pathspec-from-file='+str(d/'stage-paths.nul'),'--pathspec-file-nul');assert {x.decode() for x in git('diff','--cached','--name-only','-z').split(b'\0') if x}==allowed
msg=d/'commit-message.txt';msg.write_text("""Correct cached cardboard diagnostics and integrate resolution evidence gates

Match the FD observer to the actual production FGMRES cache mode and retain independent direct local probes. Compose volume-weighted damage, measured free-force residual checks, first-local-peak capacity identity, checked rigid-contact subarena planning and full-body host scene observers. Preserve concurrent Human cardiac source. Eight targets compile and13 integrated CPU check commands pass; a full combined native trajectory is pending.

Publish additive scope corrections for earlier uncached action measurements. Corrected cache observer599 preserves32 small-regression production files and matches cached CPU reconstruction to1.25749e-7 relative. Full strict128-Newton coupon still fails19; independent final-state projection/conditioning and direction experiments retain their limits.

Archive full contact trial geometry with12 exact transaction files; first failure is mortar complement certification and individual feature remains unresolved. Production compactor fixture598 passes parity and overflow controls without a physical trajectory. Preserve every failed/draft binding and diagnostic launch. Full solver/convergence/calibration/analogue/assembly/crush-impact qualification remains open.
""")
(d/'commit.log').write_bytes(git('commit','-F',str(msg)));head=git('rev-parse','HEAD').decode().strip();assert {x.decode() for x in git('diff-tree','--no-commit-id','--name-only','-r','-z',head).split(b'\0') if x}==allowed
(d/'push.log').write_bytes(git('push','origin','main'));remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];assert remote==head;assert not git('status','--porcelain=v1','--untracked-files=all')
receipt={'schema':'numi.cardboard.publication041-commit-push.v1','parent':a['parent'],'commit':head,'origin_main':remote,'committed_paths':len(allowed),'root_audit_sha256':sha(d/'report.json'),'summary_sha256':sha(e/'summary.json'),'binding_sha256':sha(e/'publication-binding-reference041.json'),'source_frozen_sha256':a['source_frozen_sha256'],'all_six_full_sequence_axes_open':True,'canonical_native_trajectory_pending':True,'active602_255_257_inventory_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'commit-push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
