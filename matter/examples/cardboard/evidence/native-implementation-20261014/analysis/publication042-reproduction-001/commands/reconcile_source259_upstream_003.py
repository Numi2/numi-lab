from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';old=q/'snapshots/source-integrated-globalization-contact-memory-259-002';s=q/'snapshots/source-integrated-globalization-contact-memory-259-003';d=q/'analysis/source259-upstream-reconciliation-003';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(old/'frozen-source.json')=='a9b8287bc59d44ceb71af3857762b95841a22572dbe02552c136d3a2918eb3d0'
parent='89da10e5dd7d620db154c3cea279803a112a302a';upstream='420b4f866decdfaed1f5a81ef102d3cb6efbb442'
git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
assert git('rev-parse','HEAD').decode().strip()==parent and git('rev-parse','origin/main').decode().strip()==upstream and not git('status','--porcelain=v1','--untracked-files=all')
changes=git('diff','--name-only',parent,upstream).decode().splitlines();assert len(changes)==10 and not any(n.startswith('matter/') or n=='CMakeLists.txt' for n in changes)
s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
f=json.loads((old/'frozen-source.json').read_text());cfg=json.loads((old/'configure-support-source.json').read_text());names=set(f['source_sha256'])|set(cfg['files'])
assert all(sha(old/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items())
for n in sorted(names):
 z=s/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/n,z)
overlay=[]
for n in changes:
 if n in names:
  data=git('show',upstream+':'+n);z=s/n;before=sha(z);z.write_bytes(data);overlay.append({'path':n,'before_sha256':before,'after_sha256':sha(z)})
assert sorted(x['path'] for x in overlay)==['apps/NumiHumanRestingVisual.hpp','apps/human_resting_skin_replay.mm']
f.update(snapshot_label=s.name,base_snapshot=str(old),source259_upstream_parent=upstream,source259_upstream_overlay=overlay,source_sha256={n:sha(s/n) for n in f['source_sha256']})
cfg.update(source259_upstream_parent=upstream,files={n:sha(s/n) for n in cfg['files']})
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
report={'schema':'numi.cardboard.source259-upstream-reconciliation.v1','old':str(old),'source':str(s),'upstream':upstream,'preserved_upstream_paths':changes,'overlay':overlay,'all_cardboard_bytes_exact':all(sha(s/n)==sha(old/n) for n in names if n.startswith('matter/')),'frozen_sha256':sha(s/'frozen-source.json'),'configure_support_sha256':sha(s/'configure-support-source.json'),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'script_sha256':sha(__file__),'report_sha256':sha(d/'report.json'),'source_frozen_sha256':sha(s/'frozen-source.json')},indent=2)+'\n')
print(json.dumps(report))
