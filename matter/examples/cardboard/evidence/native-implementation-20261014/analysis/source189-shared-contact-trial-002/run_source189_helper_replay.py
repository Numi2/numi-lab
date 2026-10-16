#!/usr/bin/env python3
"""Fresh, hash-gated launcher for the source189 production-helper replay.

Prepared for root-owned GPU dispatch. This file itself does not run on creation.
"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time, shlex
B=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
P=B/"analysis/source189-shared-contact-trial-002"
OLD=B/"snapshots/source-shared-contact-trial-189-buildfix002"
INPUTS=B/"analysis/contact-armijo-energy-replay-499-001/case-496-inputs-v4"
NATIVE=B/"native"
INPUT_SHA="b6c2e1872afe5f6e6685a57525fe91e604b20bf6ab5aa0fcb1acd546e80dea0f"
HOST_SHA="23f392404f9a58a221c3a505a944c21c72ec93cb960155ddfe90c885d932031c"
LIB_SHA="136bd6c922b42670852e3a6e9fbc07b2758feb702767c2704dd6bd00839ad209"
DEP_SHA="291a37ed7b1163273926492982455f1dad3275f70f3c4c7d217c70cc9d473fb6"
OUT_BYTES=8064
SRC={
 "matter/src/runtime.mm":"9ed2a0011bbecb138ac1bcb9089a07899a34b5d39b43e1cfce3b2e42565f4e01",
 "matter/src/metal/fem.metalinc":"bbd6abd764f723e6f0b266b80a3c99e100384e68e5ca5ae7cdd29cad36fb6a43",
 "matter/src/metal/contact.metalinc":"deb984d07de73477c294446eaacaae4e8a8bf374d895056018b9d8047d491cec"
}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def check_source(root):
 root=root.resolve()
 frozen=json.loads((root/"frozen-source.json").read_text())
 entries=frozen.get("source_sha256",{})
 if not isinstance(entries,dict) or not entries: raise RuntimeError("source snapshot has no source_sha256 map")
 mismatches=[]
 for rel,expected in entries.items():
  p=root/rel
  if not p.is_file() or sha(p)!=expected: mismatches.append(rel)
 if mismatches: raise RuntimeError("frozen source map mismatch: "+", ".join(mismatches[:8]))
 for rel,expected in SRC.items():
  if sha(root/rel)!=expected: raise RuntimeError("source189 source hash mismatch: "+rel)
 if sha(P/"contact_barrier_trial_replay_source189_host")!=HOST_SHA: raise RuntimeError("host hash mismatch")
 if sha(P/"contact_barrier_trial_replay_source189.metallib")!=LIB_SHA: raise RuntimeError("metallib hash mismatch")
 dep=P/"contact_barrier_trial_replay_source189.d"
 if sha(dep)!=DEP_SHA: raise RuntimeError("Metal depfile hash mismatch")
 # Rebind every source-tree dependency from the compiled 189-002 path to the
 # supplied canonical source tree and require byte identity.
 text=dep.read_text().replace("\\\n"," ")
 deps=shlex.split(text.split(":",1)[1])
 checked=0
 prefix=str(OLD)+"/"
 for item in deps:
  if item.startswith(prefix):
   rel=item[len(prefix):]
   if not (root/rel).is_file() or sha(root/rel)!=sha(pathlib.Path(item)):
    raise RuntimeError("compiled source dependency differs in supplied source root: "+rel)
   checked+=1
 if checked<10: raise RuntimeError("unexpectedly small Metal source dependency closure")
 im=INPUTS/"manifest.json"
 if sha(im)!=INPUT_SHA: raise RuntimeError("input manifest hash mismatch")
 input=json.loads(im.read_text())
 for name,row in input["files"].items():
  p=INPUTS/name
  if p.stat().st_size!=row["byte_count"] or sha(p)!=row["sha256"]:
   raise RuntimeError("input file mismatch: "+name)
 return {"source_root":str(root),"source_map_entries":len(entries),
         "compiled_source_dependency_files_rebound":checked,
         "source_hashes":SRC,"host_sha256":HOST_SHA,"metallib_sha256":LIB_SHA,
         "metal_depfile_sha256":DEP_SHA,"input_manifest_sha256":INPUT_SHA,
         "input_files":input["files"],"physical_transactions":0}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--source-root",required=True,help="canonical refreshed source189 snapshot; frozen-source.json must match")
 ap.add_argument("--output-dir",required=True,help="fresh direct child under native/")
 a=ap.parse_args()
 binding=check_source(pathlib.Path(a.source_root))
 out=pathlib.Path(a.output_dir).resolve()
 if out.parent!=NATIVE.resolve(): raise RuntimeError("output must be a direct child of native/")
 os.mkdir(out,0o700)
 (out/"preflight.json").write_text(json.dumps(binding,sort_keys=True,indent=2,allow_nan=False)+"\n")
 cmd=[str(P/"contact_barrier_trial_replay_source189_host"),
      str(P/"contact_barrier_trial_replay_source189.metallib"),str(INPUTS),str(out/"output.bin")]
 t=time.time()
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,check=False)
  (out/"stdout.log").write_text(p.stdout); (out/"stderr.log").write_text(p.stderr)
  op=out/"output.bin"
  receipt={"command":cmd,"exit_code":p.returncode,"wall_seconds":time.time()-t,
           "output_exists":op.exists(),"output_bytes":op.stat().st_size if op.exists() else None,
           "output_sha256":sha(op) if op.exists() else None,"expected_output_bytes":OUT_BYTES,
           "physical_transactions":0}
  (out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
  return 0 if p.returncode==0 and op.exists() and op.stat().st_size==OUT_BYTES else 2
 except subprocess.TimeoutExpired as e:
  (out/"stdout.log").write_text(e.stdout or ""); (out/"stderr.log").write_text(e.stderr or "")
  (out/"receipt.json").write_text(json.dumps({"command":cmd,"timeout_seconds":120,
    "status":"timeout; outputs retained","physical_transactions":0},indent=2)+"\n")
  return 124
if __name__=="__main__":
 try: sys.exit(main())
 except Exception as e:
  print("preflight_error: "+str(e),file=sys.stderr); sys.exit(2)
