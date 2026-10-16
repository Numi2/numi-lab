from pathlib import Path
import json,hashlib,difflib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-contact-compaction-fixture-249-001';src=q/'snapshots/source-primary-cache-fd-action-253-003';d=q/'analysis/source253-final-primary-cache-observer-package-004';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((src/'frozen-source.json').read_text());b=json.loads((base/'frozen-source.json').read_text())
print('frozenkeys',list(f))
key=next(k for k in ['source_files','files_sha256','files'] if isinstance(f.get(k),dict))
fm=f[key];bm=b[key];changes=[];patch=[]
def hashvalue(x):
 return x if isinstance(x,str) else x['sha256']
for name,expected in fm.items():
 p=src/name;assert sha(p)==hashvalue(expected),(name,sha(p),expected)
 if name not in bm or hashvalue(bm[name])!=hashvalue(expected):
  old=base/name;before=old.read_text().splitlines(keepends=True) if old.exists() else [];after=p.read_text().splitlines(keepends=True)
  patch.extend(difflib.unified_diff(before,after,fromfile='a/'+name,tofile='b/'+name))
  changes.append({'path':name,'before_sha256':sha(old) if old.exists() else None,'after_sha256':sha(p),'bytes':p.stat().st_size})
assert not(set(bm)-set(fm))
(d/'source253-final.patch').write_text(''.join(patch))
inputs=[src/'frozen-source.json',base/'frozen-source.json',q/'build-source253-primary-cache-action-002/source-build.json',q/'analysis/source253-build-002/report.json',q/'analysis/source253-primary-cache-fd-action-003/report.json',q/'analysis/native555-primary-cache-action-scope-erratum-253-001/report.json',q/'analysis/native555-primary-cache-action-scope-independent-audit-253-001/report.json',Path(__file__)]
report={'schema':'numi.cardboard.primary-cache-observer-final-package.v1','source':str(src.relative_to(q)),'source_map_sha256':sha(src/'frozen-source.json'),'base':str(base.relative_to(q)),'all_declared_source_entries_verified':len(fm),'changes':changes,'production_operator_and_acceptance_unchanged':True,'diagnostic_action_fixed':'FD observer now binds the actual cache and the same shared cache-mode policy as primary FGMRES; the local direct constitutive probe remains explicitly distinct.','legacy_capture_action_classification':'Explicit policy metadata absent: cached production parity is unestablished; historical numeric results retained.','CPU_validation':'source253-build-002 all10 checks PASS and source253-primary-cache-fd-action-003 four focused checks PASS.','GPU_validation':'native599 queued separately; no native result assumed here.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in d.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'patch_sha256':sha(d/'source253-final.patch'),'changed_files':len(changes),'paths':[x['path'] for x in changes]}))
