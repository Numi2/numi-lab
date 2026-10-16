from pathlib import Path
import json,hashlib,shutil,subprocess,difflib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=q/'snapshots/source-primary-cache-fd-action-253-003'
s=q/'snapshots/source-integrated-measured-coupon-contact-254-001';d=q/'analysis/source254-integrated-composition-001'
assert not s.exists() and not d.exists();s.mkdir();d.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmap(p):
 f=json.loads((p/'frozen-source.json').read_text())
 assert all(sha(p/n)==h for n,h in f['source_sha256'].items()),str(p)
 cfg=json.loads((p/'configure-support-source.json').read_text())
 assert all(sha(p/n)==h for n,h in cfg['files'].items()),str(p)
 return f,cfg
f,cfg=fmap(base);names=set(f['source_sha256'])
for n in sorted(set(cfg['files'])|names):
 z=s/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,z)
layers=[
('source248-volume-force-first-peak',q/'snapshots/source-strict-refined-contact-observers-241-004',q/'snapshots/source-shipping-first-local-peak-248-001','a9cadaa59ea11a943f914c4845d663ad6754148dd0dc58fd2007f93b9baad7a5'),
('source250-contact-arena-plan',q/'snapshots/source-strict-contact-sample-pages-247-001',q/'snapshots/source-shipping-arena-plan-250-006','72ffa7b67a9799cd8e17d8679358d29c0da38af72086c4a291846d154392c646'),
('source252-activation-host-scene-observer',q/'snapshots/source-strict-refined-contact-observers-241-004',q/'snapshots/source-freebody-scene-observer-252-002','09d96dd6f4e47ca28ac71cc4efc24884ac01f4ac9a99990c610d7599ae0c8fdf')]
records=[]
try:
 for label,ancestor,donor,expected in layers:
  af,ac=fmap(ancestor);df,dc=fmap(donor);assert sha(donor/'frozen-source.json')==expected
  changes=[n for n,h in df['source_sha256'].items() if af['source_sha256'].get(n)!=h];rows=[]
  for n in sorted(changes):
   cur=s/n;old=ancestor/n;new=donor/n
   if not cur.exists():cur.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,cur);mode='new'
   elif not old.exists():assert sha(cur)==sha(new),n;mode='identical_new'
   elif sha(cur)==sha(old):shutil.copy2(new,cur);mode='direct'
   elif sha(cur)==sha(new):mode='identical'
   else:
    r=subprocess.run(['git','merge-file','-p',str(cur),str(old),str(new)],capture_output=True)
    (d/(label+'-'+Path(n).name+'.merge.stderr')).write_bytes(r.stderr)
    if r.returncode:
     (d/(label+'-'+Path(n).name+'.conflict')).write_bytes(r.stdout)
     raise RuntimeError('three-way conflict '+label+' '+n+' code='+str(r.returncode))
    cur.write_bytes(r.stdout);mode='three_way_clean'
   names.add(n);rows.append({'path':n,'ancestor_sha256':sha(old) if old.exists() else None,'donor_sha256':sha(new),'result_sha256':sha(cur),'mode':mode})
  records.append({'label':label,'ancestor':str(ancestor),'ancestor_frozen_sha256':sha(ancestor/'frozen-source.json'),'donor':str(donor),'donor_frozen_sha256':expected,'changes':rows,'ancestor_selected_but_not_in_donor':sorted(set(af['source_sha256'])-set(df['source_sha256']))})
 excepted=False
except Exception as e:
 (d/'failed-preflight.json').write_text(json.dumps({'error':str(e),'completed_layers':records},indent=2)+'\n');raise
cfg={**cfg,'source254_support_current_overlay':True,'files':{n:sha(s/n) for n in cfg['files']}}
(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
g={**f,'snapshot_label':s.name,'base_snapshot':str(base),'scope':'Composition of corrected primary cache observer, contact-sample compaction fixture repair, volume-weighted damage, measured force accuracy, first local peak identity, fail-closed shipping arena plan and activation/free-body host-scene observers. Source composition and CPU build do not qualify full physical histories.','source254_composition':records,'source_sha256':{n:sha(s/n) for n in sorted(names)}}
(s/'frozen-source.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n')
changed=[n for n in sorted(names) if f['source_sha256'].get(n)!=sha(s/n)]
patch=[]
for n in changed:
 patch.extend(difflib.unified_diff((base/n).read_text().splitlines(keepends=True) if (base/n).exists() else [],(s/n).read_text().splitlines(keepends=True),fromfile='a/'+n,tofile='b/'+n))
(d/'source254.patch').write_text(''.join(patch))
r={'schema':'numi.cardboard.source254-composition.v1','source':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'configure_support_sha256':sha(s/'configure-support-source.json'),'base':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'source_files':len(names),'changed_files':changed,'layers':records,'GPU_execution':False,'physical_operator_changed_by_this_composition':False,'native_combined_source_qualified':False,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'script_sha256':sha(__file__),'artifacts':{p.name:sha(p) for p in d.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='layers'}))
