from pathlib import Path
import json,hashlib
BASE=Path(__file__).parent
NEW=BASE.parent/'window/native-run'
OLD=Path('/Users/n/numi-human-retained-delivery-20261009/accepted-force-observer-1253/paired-smoke-001/window/native-run')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[Path(__file__),BASE.parent/'window/run-declaration.json',BASE.parent/'window/execution.json']
rows=[]
for p in sorted(OLD.glob('*.csv')):
 other=NEW/p.name
 files.extend([p,other])
 rows.append({'name':p.name,'baseline_sha256':sha(p),'candidate_sha256':sha(other),'byte_exact':p.read_bytes()==other.read_bytes()})
packs=[]
for p in sorted(NEW.rglob('*.mrvpack')):
 old=OLD/p.relative_to(NEW)
 files.extend([p,old])
 packs.append({'path':str(p.relative_to(NEW)),'baseline_sha256':sha(old),'candidate_sha256':sha(p),'byte_exact':p.read_bytes()==old.read_bytes()})
pins={str(p):sha(p) for p in files}
exe=json.loads((BASE.parent/'window/execution.json').read_text())
report={'inputs':pins,'inputs_unchanged':all(sha(Path(p))==h for p,h in pins.items()),'native_rc':exe['returncode'],'launch_input_pins_unchanged':not exe['changed_inputs'],'argv_matches':exe['native_argv_matches_prepared_cli_preview'],'traces':rows,'packs':packs,'all_shared_traces_and_packs_exact':all(x['byte_exact'] for x in rows+packs),'scope':'1.024 s presentation change parity; no long-horizon qualification'}
(BASE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'traces':len(rows),'packs':len(packs),'all_exact':report['all_shared_traces_and_packs_exact'],'inputs_unchanged':report['inputs_unchanged']}))
assert report['all_shared_traces_and_packs_exact'] and report['inputs_unchanged'] and report['native_rc']==0
