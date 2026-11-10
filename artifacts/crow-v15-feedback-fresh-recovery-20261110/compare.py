from pathlib import Path
import gzip,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-feedback-fresh-recovery-20261110';ref=R/'artifacts/crow-v15-feedback-flight-envelope-20261110/18722003.stderr.gz';obs=A/'flight.stderr.gz'
def get(path):
 with gzip.open(path,'rt') as f:return {(x['lane'],x['step']):(x['q'],x['v']) for l in f if l.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(l.split(' ',1)[1])] if x['lane']<40 and x['step']<=800}
x=get(ref);y=get(obs);diff=[k for k in x if x[k]!=y.get(k)];h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r={'reference_count':len(x),'observer_count':len(y),'qv_differences':len(diff),'reference_sha256':h(ref),'observed_sha256':h(obs)};(A/'source-parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert len(x)==len(y)==320 and not diff
