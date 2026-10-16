from pathlib import Path
import json,hashlib,shutil,subprocess,difflib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=q/'snapshots/source-integrated-measured-coupon-contact-254-002'
s=q/'snapshots/source-integrated-globalization-contact-memory-259-001';d=q/'analysis/source259-integrated-composition-001'
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
('source255-best-of-cgls',q/'snapshots/source-primary-cache-fd-action-253-003',q/'snapshots/source-best-of-cgls-255-001','1efd6df0382dfe74e45d453fb22a81848d1fb251e99c2ab5cf6a8b8a5b062fc4'),
('source257-inline-mortar-ccd-observer',q/'snapshots/source-freebody-scene-observer-252-002',q/'snapshots/source-mortar-first-uncertified-audit-257-006','0f6d693d369c3cada323cd0b92e755fd45cb58381063f5e1c8a3117fb5ad6725'),
('source258-contact-incidence-transition-preflight',q/'snapshots/source-shipping-arena-plan-250-006',q/'snapshots/source-shipping-arena-plan-258-006','f5eff2891cd4222e894c5cf70743e11061496ebee204f94a6b156e551ad06e0e')]

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
cfg={**cfg,'source259_support_current_overlay':True,'files':{n:sha(s/n) for n in cfg['files']}}
(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
g={**f,'snapshot_label':s.name,'base_snapshot':str(base),'scope':'Composition of canonical254 with opt-in transactional Newton/CGLS selection, actual first-uncertified mortar CCD observer, exact six contact incidence allocations and fail-closed252-slot transition planner. The best-of path changes globalization only when explicitly enabled; no physical operator or acceptance tolerance is altered. Composition and CPU build do not qualify complete physical histories.','source259_composition':records,'source_sha256':{n:sha(s/n) for n in sorted(names)}}
(s/'frozen-source.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n')
changed=[n for n in sorted(names) if f['source_sha256'].get(n)!=sha(s/n)]
patch=[]
for n in changed:
 patch.extend(difflib.unified_diff((base/n).read_text().splitlines(keepends=True) if (base/n).exists() else [],(s/n).read_text().splitlines(keepends=True),fromfile='a/'+n,tofile='b/'+n))
(d/'source259.patch').write_text(''.join(patch))
r={'schema':'numi.cardboard.source259-composition.v1','source':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'configure_support_sha256':sha(s/'configure-support-source.json'),'base':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'source_files':len(names),'changed_files':changed,'layers':records,'GPU_execution':False,'physical_operator_changed_by_this_composition':False,'opt_in_globalization_changed':True,'shipping_peak_inventory_bound':False,'native_combined_source_qualified':False,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'script_sha256':sha(__file__),'artifacts':{p.name:sha(p) for p in d.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='layers'}))
