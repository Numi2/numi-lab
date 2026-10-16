from pathlib import Path
import hashlib,json,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');o=q/'analysis/source198-host-artifacts-001'
assert not o.exists();o.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
snapshots=[
'source-shared-contact-trial-189-buildfix003',
'source-damped-full-zero-cgls-190-001',
'source-memory-single-private-arena-192-001',
'source-contact-damped-cgls-memory-193-001',
'source-contact-damped-cgls-memory-195-001',
'source-damped-control-selector-197-001',
'source-contact-damped-memory-198-001']
missing=[];snaps=[]
for s in snapshots:
 p=q/'snapshots'/s
 if not p.is_dir():missing.append(s);continue
 for n in ['frozen-source.json','configure-support-source.json']:
  f=p/n
  if f.is_file():put(o/'snapshots'/s/n,f.read_bytes())
 snaps.append(s)
builds=['build-source189-003','build-source190-001','build-source190-002','build-source193-001','build-source193-002','build-source195-001','build-source197-001','build-source198-001']
rows=[];seen={}
for bn in builds:
 p=q/bn
 if not p.is_dir():missing.append(bn);continue
 for n in ['CMakeCache.txt','compile_commands.json','provenance-command-0.log','provenance-command-1.log','source-build.json']:
  f=p/n
  if f.is_file():put(o/'builds'/bn/n,f.read_bytes())
 f=p/'source-build.json'
 if not f.exists():continue
 x=json.loads(f.read_text())
 for n,h in x.get('artifacts_sha256',{}).items():
  a=Path(n);b=a.read_bytes();assert sha(b)==h,(bn,n)
  dest=Path('artifact-blobs')/h
  put(o/dest,b)
  rows.append({'build':bn,'external_path':n,'sha256':h,'bytes':len(b),'bundle_path':str(dest)})
  seen[h]=len(b)
report={'schema':'numi.cardboard.source198-host-artifacts.v1','snapshots':snaps,'missing_requested_snapshots_or_builds':missing,'builds':builds,'artifact_bindings':rows,'unique_artifacts':len(seen),'artifact_bytes':sum(seen.values()),'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Source and build preservation only; no full physical qualification.'}
(o/'collection.json').write_text(json.dumps(report,indent=2)+'\n')
(o/'collect.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in report.items() if k!='artifact_bindings'},indent=2))
