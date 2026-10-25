from pathlib import Path
import argparse,json,hashlib,gzip
p=argparse.ArgumentParser();p.add_argument("--canonical-root",type=Path,required=True);a=p.parse_args()
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((P/"canonical-manifest.json").read_text())
for r in m["canonical_paths"]:
 f=a.canonical_root/r["path"];assert sha(f.read_bytes())==r["sha256"],f
e=json.loads((P/"source-artifacts.json").read_text())
for r in e["files"]:
 b=(P/r["path"]).read_bytes();b=gzip.decompress(b)if r["gzip"]else b
 assert len(b)==r["uncompressed_size_bytes"] and sha(b)==r["uncompressed_sha256"],r["path"]
assert m["python_tests"]==34 and not m["geometry_admitted"] and m["accepted_physical_steps"]==0
print("PASS",len(m["canonical_paths"]),"canonical files;",len(e["files"]),"retained evidence files; no anatomical or physical admission")
