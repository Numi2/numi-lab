from pathlib import Path
import json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native601-full-private-trial-root-adjudication-001';d.mkdir(exist_ok=False);c=q/'native/freebody-newton7-full-private-trial-601';base=q/'native/activation-full-residual-control-on-591'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==1 and r['source_inputs_unchanged'] and not r['timed_out']
same=[];inputs=[Path(__file__),c/'receipt.json',base/'receipt.json']
for p in sorted((base/'output/dense').glob('transaction-*')):
 other=c/'output/dense'/p.name;assert sha(p)==sha(other),p.name;same.append(p.name);inputs.extend([p,other])
jp=c/'diagnostics/capture-coupled-trial-capture/capture.coupled.json';bp=jp.with_suffix('.bin');j=json.loads(jp.read_text());raw=bp.read_bytes();assert sha(bp)==j['payload']['sha256'] and len(raw)==j['payload']['byte_count']
for a in j['arrays']:
 data=raw[a['offset']:a['offset']+a['byte_count']];assert len(data)==a['byte_count'] and hashlib.sha256(data).hexdigest()==a['sha256'],a['name']
inputs.extend([jp,bp])
report={'schema':'numi.cardboard.native601-full-private-trial-root-adjudication.v1','verdict':'PASS_OBSERVER_PRODUCTION_TRANSACTION_PARITY_TARGET_STILL_FAILS','production_transaction_files_exact':same,'arrays_hash_verified':len(j['arrays']),'capture_bytes':len(raw),'capture_json_sha256':sha(jp),'capture_binary_sha256':sha(bp),'wall_seconds':r['wall_seconds'],'native_exit_code':r['exit_code'],'scope':'Same-history private trial0 before/after geometry at control1202 Newton7. All fixture transaction files are unchanged from591; the full accepted fixture is still incomplete. Individual mortar CCD feature certification is not identified by this payload.','all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},'report_sha256':sha(d/'report.json')},indent=2)+'\n');print(json.dumps(report));print(sha(d/'report.json'))
