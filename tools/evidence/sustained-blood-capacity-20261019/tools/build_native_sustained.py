#!/usr/bin/env python3
import hashlib,json,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
evidence=root/"native-sustained-capacity-001"
previous=json.loads((evidence/"build-013/build-receipt.json").read_text())
snapshot=evidence/"build-014"
snapshot.mkdir(parents=True,exist_ok=True)
for rel in ["HumanRespirationSustained.metal","sustained_capacity_probe.mm"]:shutil.copy2(evidence/rel,snapshot/rel)
for rel in ["source/matter/include/numi/matter/human_respiration.h","source/matter/src/human_respiration.metal","source/matter/tools/human_resting_runtime.hpp"]:shutil.copy2(root/rel,snapshot/Path(rel).name)
replace=lambda argv:[x.replace("/build-010/","/build-010/") for x in argv]
compile_argv=replace(previous["compile_argv"])
link_argv=replace(previous["link_argv"])
start=time.monotonic();c=subprocess.run(compile_argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);compile_s=time.monotonic()-start
(snapshot/"compile.log").write_text(c.stdout)
if c.returncode:print(c.stdout);raise SystemExit(c.returncode)
start=time.monotonic();l=subprocess.run(link_argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);link_s=time.monotonic()-start
(snapshot/"link.log").write_text(l.stdout)
if l.returncode:print(l.stdout);raise SystemExit(l.returncode)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lib=Path("/Users/n/numi-human-separation-20261018/C/deformable-contact-finalizer-233/build/libmetalrobo.dylib")
receipt={"schema":"numi.sustained_blood_native_build.v1","source_checkout":str(root),"compile_argv":compile_argv,"compile_status":c.returncode,"compile_seconds":compile_s,"link_argv":link_argv,"link_status":l.returncode,"link_seconds":link_s,"binary_sha256":sha(evidence/"sustained_capacity_probe"),"shader_source":str(evidence/"HumanRespirationSustained.metal"),"shader_source_sha256":sha(evidence/"HumanRespirationSustained.metal"),"flatten_receipt_sha256":sha(evidence/"shader-flatten-receipt.json"),"mpm_shader_sha256":sha(root/"source/matter/src/human_respiration.metal"),"probe_source_sha256":sha(evidence/"sustained_capacity_probe.mm"),"runtime_header_sha256":sha(root/"source/matter/tools/human_resting_runtime.hpp"),"runtime_library":str(lib),"runtime_library_sha256":sha(lib)}
(snapshot/"build-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"PASS","build":str(snapshot),"binary_sha256":receipt["binary_sha256"],"compile_seconds":compile_s,"link_seconds":link_s},indent=2))
