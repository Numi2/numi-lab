#!/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13
"""Seal static host-bound checker inputs and outputs after validation."""
from pathlib import Path
import hashlib,json,datetime
P=Path(__file__).resolve().parent
rows={}
for f in sorted(P.rglob('*')):
    if f.is_symlink(): raise SystemExit(f'symlink forbidden: {f}')
    if not f.is_file(): continue
    rel=f.relative_to(P)
    if rel.as_posix()=='package-manifest.json' or rel.parts[0]=='executions' or '__pycache__' in rel.parts: continue
    h=hashlib.sha256(f.read_bytes()).hexdigest()
    rows[rel.as_posix()]={'bytes':f.stat().st_size,'sha256':h}
manifest={'schema':'numi.cardboard.sparse-scan-macmini-package.v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'macmini','source170_checker_sha256':rows['source/cardboard_sparse_private_contact_scan_native_check.mm']['sha256'],'source170_o3_metallib_sha256':rows['NumiMatter-O3.metallib']['sha256'],'fresh_checker_binary_sha256':rows['validation/sparse-scan-checker']['sha256'],'native_gpu_case_count':49,'total_environment_pair_slots':6366845,'gpu_dispatch_occurred_during_package_preparation':False,'files':rows}
out=P/'package-manifest.json'
out.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
print(json.dumps({'manifest':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'file_count':len(rows)},sort_keys=True))
