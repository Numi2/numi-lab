#!/usr/bin/env python3
"""Restore the archived cued training runtime, sources and checkpoint from Git."""
import importlib.util,json,os,sys
from pathlib import Path
root=Path.cwd();out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location("archive",root/"tools/crow_evidence_archive.py")
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
entries=tool.tree();base="artifacts/crow-rl-optimizer-audit-20261108/"
records=[]
with tool.Objects() as objects:
 metadata=json.loads(objects.read(entries[base+"runtime-recovery.json"][1]))
 manifests={}
 for name in metadata["manifests"]:
  manifests[name]={f["path"]:f for f in json.loads(objects.read(entries[name][1]))["files"]}
 def save(relative,data,sha,mode):
  dest=out/tool.safe_path(relative);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);dest.chmod(mode)
  if tool.file_hash(dest)!=sha:raise RuntimeError("Git runtime readback mismatch: "+relative)
  records.append({"path":relative,"sha256":sha,"bytes":len(data),"mode":mode})
 for relative,record in metadata["native"].items():
  f=next(m[record["archive_source"]] for m in manifests.values() if record["archive_source"] in m)
  save(relative,b"".join(tool.content(f,entries,objects)),record["sha256"],f["mode"])
 for path,record in metadata["python_sources"].items():
  save(path,objects.read(record["git_blob"]),record["sha256"],record["mode"])
 for path,record in metadata["git_inputs"].items():
  save(path,objects.read(record["git_blob"]),record["sha256"],record["mode"])
 for relative,source in metadata["checkpoint"].items():
  f=next(m[source] for m in manifests.values() if source in m)
  save(relative,b"".join(tool.content(f,entries,objects)),f["sha256"],f["mode"])
tool.write_json(out/"runtime-restoration.json",{"status":"passed","archive_commit":tool.git("rev-parse","HEAD").decode().strip(),"files":records,
 "scope":"Exact Git blob recovery and filesystem readback; matching Python/MLX/NumPy, macOS and native device remain required."})
print(json.dumps({"status":"passed","restored_files":len(records),"restored_bytes":sum(f["bytes"] for f in records)}))
