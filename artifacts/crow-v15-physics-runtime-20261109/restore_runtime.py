from pathlib import Path
import gzip,hashlib,json,sys
A=Path(__file__).resolve().parent
if len(sys.argv)!=2:raise SystemExit('usage: python3 restore_runtime.py NEW_EMPTY_DIRECTORY')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
for row in json.loads((A/'runtime.json').read_text())['components']:
 data=gzip.decompress((A/row['archive']).read_bytes());assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['uncompressed_sha256'];name=row['archive'].removesuffix('.gz');dest=out/name
 if name=='materialized-late-libmetalrobo.dylib':dest=out/'libmetalrobo.dylib'
 dest.write_bytes(data)
 if name=='metalrobo_task_rollout':dest.chmod(0o755)
 print(dest,hashlib.sha256(data).hexdigest())
