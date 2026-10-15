from pathlib import Path
import gzip,hashlib,json,shutil
base=Path('/Users/home/cardboard-implementation-20261014')
output=base/'analysis/retained-restart-captures100-001'
output.mkdir(exist_ok=False)
sha=lambda data:hashlib.sha256(data).hexdigest()
rows=[]
for number in range(335,343):
 command_path=base/f'native-{number}-command.json'
 command=json.loads(command_path.read_text())
 case=Path(command[command.index('--case')+1])
 receipt=json.loads((case/'receipt.json').read_text())
 assert receipt['source_inputs_unchanged'] and receipt['exit_code']==2 and not receipt['timed_out']
 raw_path=case/'operator-capture'
 raw=raw_path.read_bytes();compressed=gzip.compress(raw,mtime=0)
 restored=gzip.decompress(compressed);assert restored==raw
 folder=output/str(number);folder.mkdir()
 (folder/'operator-capture.gz').write_bytes(compressed)
 for source,name in ((command_path,'native-command.json'),(case/'prediction.json','prediction.json'),(case/'receipt.json','receipt.json'),(case/'inputs/external/source-build.json','source-build.json')):
  shutil.copy2(source,folder/name)
 conditioning=case/'output/rejected-conditioning-000001'
 shutil.copytree(conditioning,folder/'conditioning')
 rows.append(dict(run=number,case=str(case),raw_source=str(raw_path),uncompressed_bytes=len(raw),uncompressed_sha256=sha(raw),compressed=f'{number}/operator-capture.gz',compressed_bytes=len(compressed),compressed_sha256=sha(compressed),restoration_verified=True,conditioning_manifest=f'{number}/conditioning/manifest.json',conditioning_manifest_sha256=sha((conditioning/'manifest.json').read_bytes())))
report=dict(schema='numi.cardboard.retained_restart_capture_bundle.v1',source_snapshot=str(base/'snapshots/source-exact-tiling-inactive-fields-100'),captures=rows,accepted_checkpoint=False,physical_convergence=False,qualification='Lossless observational diagnostics from strict failed terminal certificates. Source100 contains a known contact regression; no production promotion or augmentation validity follows from capture restoration.')
(output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
shutil.copy2(Path(__file__),output/'seal.py')
print(json.dumps(dict(captures=len(rows),raw_bytes=sum(r['uncompressed_bytes'] for r in rows),compressed_bytes=sum(r['compressed_bytes'] for r in rows))))
