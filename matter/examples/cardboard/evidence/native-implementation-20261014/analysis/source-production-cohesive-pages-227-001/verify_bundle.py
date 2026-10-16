#!/usr/bin/env python3
import os,json,hashlib,sys
BASE="/Users/n/cardboard-implementation-20261014-host-macmini"
SNAP=BASE+"/snapshots/source-production-cohesive-pages-227-001"
PKG=BASE+"/analysis/source-production-cohesive-pages-227-001"
BUILD=BASE+"/build-source-production-cohesive-pages-227-001"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
frozen=json.load(open(SNAP+"/frozen-source.json"))
listed=frozen["source_sha256"]
parent="/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-shipping-contact-pages-225-002"
parent_map=json.load(open(parent+"/frozen-source.json"))["source_sha256"]
parent_actual={}
for here,dirs,files in os.walk(parent):
    dirs.sort()
    for n in sorted(files):
        p=os.path.join(here,n); rel=os.path.relpath(p,parent).replace(os.sep,"/")
        if rel=="frozen-source.json": continue
        parent_actual[rel]=sha(p)
assert parent_actual==parent_map
actual={}
for here,dirs,files in os.walk(SNAP):
 dirs.sort()
 for n in sorted(files):
  p=os.path.join(here,n); rel=os.path.relpath(p,SNAP).replace(os.sep,"/")
  if rel=="frozen-source.json": continue
  actual[rel]=sha(p)
assert actual==listed, (len(actual),len(listed),len(set(actual)^set(listed)))
overlay=json.load(open(PKG+"/source-overlay.json"))
assert sha(PKG+"/overlay.patch")==overlay["patch_sha256"]
assert sha(SNAP+"/frozen-source.json")==overlay["frozen_source_sha256"]
assert len(overlay["changed_files"])==7
assert sha(PKG+"/source-build.json")==overlay["source_build_record_sha256"]
assert sha(PKG+"/patch-dry-run.log")==overlay["patch_dry_run_log"]["sha256"]
parent="/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-shipping-contact-pages-225-002"
actual_delta={}
for rel,entry in overlay["changed_files"].items():
    assert (sha(parent+"/"+rel) if os.path.exists(parent+"/"+rel) else None)==entry["before_sha256"],rel
    actual_delta[rel]=sha(SNAP+"/"+rel)
assert set(actual_delta)==set(overlay["changed_files"])
for rel,entry in overlay["changed_files"].items():
 assert sha(SNAP+"/"+rel)==entry["after_sha256"],rel
buildrec=json.load(open(PKG+"/source-build.json"))
for rel,entry in buildrec["artifacts"].items():
 assert sha(BUILD+"/"+rel)==entry["sha256"],rel
assert buildrec["frozen_source_sha256"]==overlay["frozen_source_sha256"]
manifest_path=PKG+"/manifest.json"
if os.path.exists(manifest_path):
    manifest=json.load(open(manifest_path))
    recorded={x["path"]:x for x in manifest["files"]}
    package_actual={}
    for n in sorted(os.listdir(PKG)):
        if n=="manifest.json": continue
        q=PKG+"/"+n
        if os.path.isfile(q): package_actual[n]={"path":n,"bytes":os.path.getsize(q),"sha256":sha(q)}
    assert recorded==package_actual
    assert manifest["snapshot_frozen_source_sha256"]==overlay["frozen_source_sha256"]
print(json.dumps({"frozen_files":len(actual),"changed_files":len(overlay["changed_files"]),"patch_sha256":overlay["patch_sha256"],"frozen_sha256":overlay["frozen_source_sha256"],"build_artifacts_verified":len(buildrec["artifacts"]),"result":"PASS"},sort_keys=True))
