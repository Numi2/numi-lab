#!/usr/bin/env python3
import hashlib, json, pathlib, sys
Q = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG = Q / "analysis/native-damped-cgls-helper-replay-190-002"
SRC = Q / "snapshots/source-damped-full-zero-cgls-190-001"
BUILD = Q / "build-source190-002"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

validation=json.loads((PKG/"build/validation.json").read_text())
if validation["result"]!="PASS" or validation["gpu_dispatch_performed"] or validation["helper_executable_invoked"]:
    raise SystemExit("validation report is not compile-only PASS")
frozen=SRC/"frozen-source.json"
build_meta=BUILD/"source-build.json"
frozen_doc=json.loads(frozen.read_text())
base_cgls=SRC/"matter/src/metal/base_cgls.metalinc"
if sha(frozen)!="1189d65c3d7f29d326155eee687979c8dd5fc1c79360a333bd569a08ad30dde7":
    raise SystemExit("source-190 frozen manifest changed")
if frozen_doc["source_sha256"].get("matter/src/metal/base_cgls.metalinc") != sha(base_cgls):
    raise SystemExit("production CGLS source does not match frozen source manifest")
files=[]
for p in sorted(PKG.rglob("*")):
    if p.is_symlink():
        raise SystemExit(f"symlink rejected: {p}")
    if p.is_file() and p.name not in {"package-manifest.json","package-manifest.sha256"}:
        files.append({"path":p.relative_to(PKG).as_posix(),"size":p.stat().st_size,"sha256":sha(p)})
manifest={
  "schema":"numi.cardboard.damped-cgls-helper-replay-package.v1",
  "scope":"Compile-only helper replay preparation; no native checker execution or GPU dispatch.",
  "package_root":str(PKG),
  "source_snapshot":str(SRC),
  "source_frozen_manifest_sha256":sha(frozen),
  "source_build_manifest":str(build_meta),
  "source_build_manifest_sha256":sha(build_meta),
  "production_cgls_source":{"path":str(base_cgls),"sha256":sha(base_cgls)},
  "fixture_metal_sha256":sha(PKG/"source/damped_cgls_fixture.metal"),
  "host_checker_sha256":sha(PKG/"source/damped_cgls_helper_replay.mm"),
  "cpu_oracle_sha256":sha(PKG/"source/cpu_oracle.py"),
  "validation_report_sha256":sha(PKG/"build/validation.json"),
  "native_gpu_executed":False,
  "helper_executable_invoked":False,
  "result":"PASS_COMPILE_ONLY",
  "files":files
}
out=PKG/"package-manifest.json"
out.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
digest=sha(out)
(PKG/"package-manifest.sha256").write_text(f"{digest}  package-manifest.json\n")
print(json.dumps({"result":manifest["result"],"file_count":len(files),"manifest_sha256":digest},sort_keys=True))
