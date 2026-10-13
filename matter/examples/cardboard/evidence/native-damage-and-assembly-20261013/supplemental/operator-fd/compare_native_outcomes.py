from pathlib import Path
import hashlib,json,datetime
r=Path('/Users/home/cardboard-followup-20261013')
a=r/'solver/operator-fd-baseline-001'; b=r/'solver/operator-fd-001'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert (a/'receipt.json').is_file() and (b/'receipt.json').is_file()
pa=json.loads((a/'prediction.json').read_text());pb=json.loads((b/'prediction.json').read_text())
verified={}
for case,pred in ((a,pa),(b,pb)):
 hashes=pred['inputs_sha256']; bad=[str(k) for k,v in hashes.items() if sha(case/'inputs'/k)!=v]
 verified[str(case.relative_to(r))]={'input_count':len(hashes),'invalid':bad}
mesh_a={str(p.relative_to(a/'output')):sha(p) for p in (a/'output').rglob('*.obj')}
mesh_b={str(p.relative_to(b/'output')):sha(p) for p in (b/'output').rglob('*.obj')}
trace=[]
for line in (b/'stderr.log').read_text().splitlines():
 if line.startswith('fem_operator_fd_iteration='): trace.append(json.loads(line.split('=',1)[1]))
result={'schema':'numi.cardboard.operator-fd-outcome-comparison.v1','derived_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limitation':'Post-execution comparison; not an original prediction seal, material oracle or convergence admission.','baseline_receipt':json.loads((a/'receipt.json').read_text()),'diagnostic_receipt':json.loads((b/'receipt.json').read_text()),'sealed_input_verification':verified,'sealed_input_maps_identical':pa['inputs_sha256']==pb['inputs_sha256'],'csv_byte_identical':(a/'output/observations.csv').read_bytes()==(b/'output/observations.csv').read_bytes(),'mesh_file_count_baseline':len(mesh_a),'mesh_file_count_diagnostic':len(mesh_b),'all_mesh_hashes_identical':mesh_a==mesh_b,'mesh_hashes_baseline':mesh_a,'mesh_hashes_diagnostic':mesh_b,'result_json_byte_identical':(a/'output/result.json').read_bytes()==(b/'output/result.json').read_bytes(),'compiled_pack_byte_identical':sha(a/'output/compiled.nmatterpack')==sha(b/'output/compiled.nmatterpack'),'fd_traces':trace}
out=r/'operator-fd/outcome-comparison.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('mesh_hashes_baseline','mesh_hashes_diagnostic')},indent=2))
