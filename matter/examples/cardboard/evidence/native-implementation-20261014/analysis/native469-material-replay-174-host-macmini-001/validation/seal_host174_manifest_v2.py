#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,pathlib,stat
PKG=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini/analysis/native469-material-replay-174-host-macmini-001')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=PKG/'manifest.json'
 if manifest.exists():raise SystemExit('manifest already exists; refusing overwrite')
 if not (PKG/'validation/host-build.json').is_file():raise SystemExit('host build record missing')
 prepath=PKG/'validation/preflight-v3-summary.json'
 if not prepath.is_file():raise SystemExit('v3 persisted preflight summary missing')
 pre=json.loads(prepath.read_text())
 if pre.get('gpu_execution') is not False or pre.get('cpu_admission',{}).get('status')!='pass':raise SystemExit('preflight record does not prove CPU-only admission')
 if (PKG/'runs/native469-source173-material-replay/output').exists():raise SystemExit('native output exists; refusing a no-native seal')
 rows=[]
 for p in sorted(PKG.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(PKG).as_posix()
  if rel=='manifest.json':continue
  if '__pycache__' in p.parts or p.suffix=='.pyc':raise SystemExit(f'cache file found: {rel}')
  st=p.stat()
  rows.append({'path':rel,'bytes':st.st_size,'sha256':sha(p),'executable':bool(st.st_mode & stat.S_IXUSR)})
 result={'schema':'numi.cardboard.native469-host-macmini-package.v1','package':str(PKG),'status':'host_build_and_cpu_admission_pass_native_replay_not_executed','manifest_self_excluded':True,'native_output_exists':False,'preflight_result_sha256':sha(prepath),'source_overlay_sha256':sha(PKG/'source-overlay.json'),'file_count':len(rows),'total_bytes':sum(x['bytes'] for x in rows),'files':rows}
 manifest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'file_count':result['file_count'],'total_bytes':result['total_bytes'],'manifest_sha256':sha(manifest)},indent=2))
if __name__=='__main__':main()
