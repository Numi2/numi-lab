#!/usr/bin/env python3
import os,json,hashlib
BASE="/Users/n/cardboard-implementation-20261014-host-macmini"
PARENT=BASE+"/snapshots/source-production-cohesive-pages-227-001"
SNAP=BASE+"/snapshots/source-production-cohesive-pages-227-002"
PKG=BASE+"/analysis/source-production-cohesive-pages-227-002"
BUILD=BASE+"/build-source-production-cohesive-pages-227-002"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def tree_hashes(root,skip_frozen=True):
 out={}
 for here,dirs,names in os.walk(root):
  dirs.sort()
  for n in sorted(names):
   p=os.path.join(here,n); rel=os.path.relpath(p,root).replace(os.sep,"/")
   if skip_frozen and rel=="frozen-source.json":continue
   out[rel]=sha(p)
 return out
frozen=json.load(open(SNAP+"/frozen-source.json"))
assert tree_hashes(SNAP)==frozen["source_sha256"]
parent_frozen=json.load(open(PARENT+"/frozen-source.json"))
assert tree_hashes(PARENT)==parent_frozen["source_sha256"]
ov=json.load(open(PKG+"/source-overlay.json"))
assert sha(SNAP+"/frozen-source.json")==ov["frozen_source_sha256"]
assert sha(PKG+"/overlay.patch")==ov["patch_sha256"]
assert sha(PKG+"/patch-dry-run.log")==ov["patch_dry_run_log"]["sha256"]
assert sha(PKG+"/source-build.json")==ov["source_build_record_sha256"]
for rel,entry in ov["changed_files"].items():
 assert sha(PARENT+"/"+rel)==entry["before_sha256"]
 assert sha(SNAP+"/"+rel)==entry["after_sha256"]
r=json.load(open(PKG+"/source-build.json"))
for rel,entry in r["artifacts"].items(): assert sha(BUILD+"/"+rel)==entry["sha256"]
assert r["frozen_source_sha256"]==ov["frozen_source_sha256"]
assert "build_log" in r and r["build_log"]["exit_code"]==0
m=PKG+"/manifest.json"
if os.path.exists(m):
 recorded={x["path"]:x for x in json.load(open(m))["files"]}
 actual={}
 for n in sorted(os.listdir(PKG)):
  if n=="manifest.json":continue
  p=PKG+"/"+n
  if os.path.isfile(p):actual[n]={"path":n,"bytes":os.path.getsize(p),"sha256":sha(p)}
 assert actual==recorded
print(json.dumps({"result":"PASS","source_count":len(frozen["source_sha256"]),"parent_source_count":len(parent_frozen["source_sha256"]),"changed_paths":len(ov["changed_files"]),"artifacts":len(r["artifacts"]),"frozen_source_sha256":ov["frozen_source_sha256"],"patch_sha256":ov["patch_sha256"]},sort_keys=True))
