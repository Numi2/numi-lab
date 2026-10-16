#!/usr/bin/env python3
import hashlib, json, os, pathlib, platform, subprocess, sys
Q = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG = Q / "analysis/native-damped-cgls-helper-replay-190-002"
SRC = Q / "snapshots/source-damped-full-zero-cgls-190-001"
BUILD = Q / "build-source190-002"
OUT = PKG / "build"
PYTHON = "/opt/homebrew/opt/python@3.13/bin/python3.13"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def run(name, argv):
    cp=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log=OUT / name
    log.write_text(cp.stdout)
    return {"argv":argv,"exit_code":cp.returncode,"log":str(log.relative_to(PKG)),"log_sha256":sha(log)}

commands=[]
metal="/usr/bin/xcrun"
for opt in ("O0","O3"):
    d=OUT/opt; d.mkdir(parents=True,exist_ok=True)
    air=d/"damped_cgls_fixture.air"; lib=d/"damped_cgls_fixture.metallib"; exe=d/"damped_cgls_helper_replay"
    commands.append(run(f"{opt}/metal-compile.log",[metal,"-sdk","macosx","metal","-std=metal4.0",f"-{opt}","-fno-fast-math","-I",str(SRC/"matter/include"),"-I",str(SRC/"include"),"-I",str(SRC/"matter/src"),"-I",str(SRC/"matter/src/metal"),"-c",str(PKG/"source/damped_cgls_fixture.metal"),"-o",str(air)]))
    commands.append(run(f"{opt}/metallib-link.log",[metal,"-sdk","macosx","metallib",str(air),"-o",str(lib)]))
    commands.append(run(f"{opt}/host-compile.log",[metal,"-sdk","macosx","clang++","-std=c++17",f"-{opt}","-fobjc-arc","-Wall","-Wextra","-Werror",str(PKG/"source/damped_cgls_helper_replay.mm"),"-framework","Metal","-framework","Foundation","-o",str(exe)]))

oracle=subprocess.run([PYTHON,str(PKG/"source/cpu_oracle.py")],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(OUT/"cpu-oracle.json").write_text(oracle.stdout)
commands.append({"argv":[PYTHON,str(PKG/"source/cpu_oracle.py")],"exit_code":oracle.returncode,"log":"build/cpu-oracle.json","log_sha256":sha(OUT/"cpu-oracle.json")})

frozen=SRC/"frozen-source.json"
frozen_doc=json.loads(frozen.read_text())
source_checks={}
for rel,expected in frozen_doc["source_sha256"].items():
    p=SRC/"matter"/rel if rel.startswith(("src/","include/","cmake/")) or rel=="CMakeLists.txt" else None
    # The frozen manifest is rooted at the repository, while this snapshot is a
    # wider tree. Resolve repository-relative paths directly from SRC.
    p=SRC/rel
    if p.is_file():
        actual=sha(p); source_checks[rel]={"expected":expected,"actual":actual,"ok":actual==expected}

build_doc=json.loads((BUILD/"source-build.json").read_text())
artifact_checks={}
for path,expected in build_doc["artifacts_sha256"].items():
    p=pathlib.Path(path)
    artifact_checks[str(p)]= {"expected":expected,"actual":sha(p) if p.is_file() else None,"ok":p.is_file() and sha(p)==expected}

errors=[]
if not all(c["exit_code"]==0 for c in commands): errors.append("compile_or_cpu_oracle_command_failed")
if not frozen_doc.get("schema")=="numi.cardboard.frozen-source.v1": errors.append("unexpected_frozen_source_schema")
if not build_doc.get("success"): errors.append("source190_build_not_successful")
if not all(v["ok"] for v in source_checks.values()): errors.append("source_hash_mismatch")
if not all(v["ok"] for v in artifact_checks.values()): errors.append("source190_build_artifact_mismatch")
if oracle.returncode or json.loads(oracle.stdout).get("result")!="PASS": errors.append("cpu_oracle_failed")
for opt in ("O0","O3"):
    for f in ("damped_cgls_fixture.air","damped_cgls_fixture.metallib","damped_cgls_helper_replay"):
        if not (OUT/opt/f).is_file() or (OUT/opt/f).stat().st_size==0: errors.append(f"missing_{opt}_{f}")
report={"schema":"numi.cardboard.damped-cgls-helper-compile-validation.v1","host":platform.platform(),"source_snapshot":str(SRC),"frozen_source_sha256":sha(frozen),"source_build_manifest":str(BUILD/"source-build.json"),"source_build_manifest_sha256":sha(BUILD/"source-build.json"),"source_build_success":build_doc.get("success"),"source_file_hash_checks":source_checks,"source_build_artifact_checks":artifact_checks,"commands":commands,"gpu_dispatch_performed":False,"helper_executable_invoked":False,"native_kernel_replay_result":None,"result":"PASS" if not errors else "FAIL","errors":errors}
(OUT/"validation.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"result":report["result"],"commands":len(commands),"source_files_checked":len(source_checks),"source_artifacts_checked":len(artifact_checks),"errors":errors},sort_keys=True))
sys.exit(0 if not errors else 1)
