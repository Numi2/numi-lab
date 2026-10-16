from pathlib import Path
import json,hashlib,shutil,gzip,stat,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e;a=q/'analysis/publication040-candidate-001';d=q/'analysis/publication040-reproduction-support-004';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();enc=lambda x:(json.dumps(x,indent=2)+'\n').encode()
s=q/'snapshots/source-strict-refined-contact-observers-241-004';cfg=json.loads((s/'configure-support-source.json').read_text())['files'];exceptions=[]
for n,h in cfg.items():
 p=r/n
 if p.is_file() and sha(p)==h:continue
 p=d/'canonical-support'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/n,p);assert sha(p)==h
 exceptions.append({'name':n,'bytes':p.stat().st_size,'sha256':h})
assert exceptions==[{'name':'build-cardboard-native-worker/worker.lock','bytes':0,'sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}]
commands={}
for n in ['audit_publication040_002.py','publish_publication040_002.py','extend_publication040_support_004.py']:
 p=q/n;z=d/'commands'/n;z.parent.mkdir(exist_ok=True);shutil.copy2(p,z);commands[n]=sha(p)
(d/'README.md').write_text("""Reference040 additive support and audit continuation

Canonical configure-support binds one zero-byte historical worker.lock beyond the authored checkout files. This package stores that exact zero-byte snapshot file for reconstruction; it is not an active worker ownership claim and is not copied into the live worker directory.

Root audit001 failed because it assumed every support file was authored in the checkout. Audit002 instead verifies every selected source in the checkout and each support input in either the checkout or this explicit exception package. No production source, binary, physical history or acceptance criterion changed.

Use audit_publication040_002.py and publish_publication040_002.py to continue this prepared candidate. The prior publication scripts remain preserved as unexecuted/superseded provenance.
""")
(d/'report.json').write_bytes(enc({'schema':'numi.cardboard.publication040-support-extension.v1','canonical_support_exceptions':exceptions,'commands':commands,'physical_results_changed':False}))
(d/'manifest.json').write_bytes(enc({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}}))
summary=json.loads((dest/'summary.json').read_text());binding=json.loads((dest/'publication-binding-reference040.json').read_text());meta=json.loads((a/'report.json').read_text());selection=json.loads((dest/'publication040-case-selection.json').read_text())
assert len(summary['records'])==2142 and meta['new_records']==80
allowed={x.decode() for x in (a/'stage-paths.nul').read_bytes().split(b'\0') if x}
for root in [d,q/'analysis/publication040-root-audit-001']:
 case=str(root.relative_to(q));assert case not in selection['cases'];fs=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink();rel=str(p.relative_to(root));raw=p.read_bytes();stored=Path(case)/rel;gz=len(raw)>262144 and p.suffix!='.gz';data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
  if gz:stored=stored.with_name(stored.name+'.gz')
  target=dest/stored;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
  assert (gzip.decompress(target.read_bytes()) if gz else target.read_bytes())==raw
  item={'name':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'external_path':str(p),'mode_octal':oct(mode)}
  if gz:item.update({'repository_compressed_copy':str(stored),'compressed_bytes':len(data),'compressed_sha256':sha(target),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
  else:item.update({'repository_copy':str(stored),'restore':'copy exact bytes and restore recorded mode'})
  fs.append(item);allowed.add(str(e/stored))
 summary['records'].append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs});selection['cases'].append(case);meta['payloads']+=len(fs)
meta['new_records']+=2;binding['new_records']+=2;selection['cases'].sort()
(dest/'summary.json').write_bytes(enc(summary));(dest/'publication040-case-selection.json').write_bytes(enc(selection))
binding['summary_sha256']=sha(dest/'summary.json');binding['case_selection_sha256']=sha(dest/'publication040-case-selection.json');binding['support_exception_package']='analysis/publication040-reproduction-support-004';(dest/'publication-binding-reference040.json').write_bytes(enc(binding))
meta['paths']=len(allowed);meta['summary_sha256']=sha(dest/'summary.json');meta['binding_sha256']=sha(dest/'publication-binding-reference040.json');meta['support_extension_added_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(a/'report.json').write_bytes(enc(meta))
(a/'stage-paths.nul').write_bytes(b''.join(x.encode()+b'\0' for x in sorted(allowed)))
print(json.dumps({'new_records':meta['new_records'],'paths':meta['paths'],'summary_sha256':meta['summary_sha256'],'binding_sha256':meta['binding_sha256']}))
