from pathlib import Path
import gzip,hashlib,json,shutil
b=Path('/Users/home/cardboard-implementation-20261014');o=b/'analysis/retained-operator-contact-captures101-001';o.mkdir(exist_ok=False);sha=lambda x:hashlib.sha256(x).hexdigest();rows=[]
for n in (344,345,346,348):
 command_path=b/f'native-{n}-command.json';cmd=json.loads(command_path.read_text());r=Path(cmd[cmd.index('--case')+1]);receipt=json.loads((r/'receipt.json').read_text());assert receipt['source_inputs_unchanged']and not receipt['timed_out']and receipt['exit_code']in(1,2);target=o/str(n);target.mkdir()
 if n in(344,345):rawpath=r/'operator-capture';capturemanifest=None
 else:
  capturemanifest=r/('private-trial.coupled.json'if n==346else'center-capture.coupled.json');m=json.loads(capturemanifest.read_text());rawpath=Path(m['payload']['file']);raw=rawpath.read_bytes();assert sha(raw)==m['payload']['sha256']and len(raw)==m['payload']['byte_count'];assert len(m['arrays'])==m['array_count'];end=0
  for a in m['arrays']:
   assert a['offset']==end and a['byte_count']==a['count']*a['element_bytes'];end+=a['byte_count'];assert sha(raw[a['offset']:end])==a['sha256']
  assert end==len(raw);shutil.copy2(capturemanifest,target/capturemanifest.name)
 raw=rawpath.read_bytes();gz=gzip.compress(raw,mtime=0);assert gzip.decompress(gz)==raw;(target/(rawpath.name+'.gz')).write_bytes(gz)
 for source,name in [(command_path,'native-command.json'),(r/'prediction.json','prediction.json'),(r/'receipt.json','receipt.json'),(r/'inputs/external/source-build.json','source-build.json')]:shutil.copy2(source,target/name)
 if n in(344,345):shutil.copytree(r/'output/rejected-conditioning-000001',target/'conditioning')
 rows.append(dict(run=n,case=str(r),raw_source=str(rawpath),uncompressed_bytes=len(raw),uncompressed_sha256=sha(raw),compressed=f'{n}/{rawpath.name}.gz',compressed_bytes=len(gz),compressed_sha256=sha(gz),restore_verified=True,array_count=None if capturemanifest is None else m['array_count']))
report=dict(schema='numi.cardboard.retained_diagnostic_bundle.v1',source_snapshot=str(b/'snapshots/source-boundary-reset-order-101'),captures=rows,accepted_checkpoint=False,qualification='Lossless strict failed-run diagnostics; no complete physical trajectory.344 admits currentNewtonaugmentationbutdoesnotacceptstep.346/348selectedNewton12comesafterearlierArmijorejection; cannotcomparethatstatewith099Newtonsasidenticalcandidate.',restoration='gzip decompress then check each uncompressed byte extent and SHA256; adjacent original capture manifests remain unmodified')
(o/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(Path(__file__),o/'seal.py');print(json.dumps(dict(captures=len(rows),raw_bytes=sum(r['uncompressed_bytes']for r in rows),compressed_bytes=sum(r['compressed_bytes']for r in rows))))
