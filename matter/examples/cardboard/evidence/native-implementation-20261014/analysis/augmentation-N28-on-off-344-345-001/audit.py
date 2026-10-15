from pathlib import Path
import json,hashlib
b=Path('/Users/home/cardboard-implementation-20261014'); out=b/'analysis/augmentation-N28-on-off-344-345-001';out.mkdir(exist_ok=False)
runs=[b/'solver/reset-order-augmentation-on-N28-capture27-344',b/'solver/reset-order-augmentation-off-N28-capture27-345']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def trace(p):
 prefix='fem_residual_armijo_iteration='
 return [json.loads(l.split(prefix,1)[1])for l in(p/'stderr.log').read_text().splitlines()if l.startswith(prefix)]
a,c=map(trace,runs);assert len(a)==len(c)==28
names=[set(p.name for p in(r/'output').iterdir()if p.is_file()and not p.name.endswith('.log'))for r in runs];assert names[0]==names[1]
files=[dict(name=n,sha256_on=h(runs[0]/'output'/n),sha256_off=h(runs[1]/'output'/n),identical=h(runs[0]/'output'/n)==h(runs[1]/'output'/n))for n in sorted(names[0])]
report=dict(schema='numi.cardboard.augmentation-comparison.v2',runs=list(map(str,runs)),receipts=[json.loads((r/'receipt.json').read_text())for r in runs],armijo_count=28,identical_before_first_attempt=a[:27]==c[:27],changed_newtons=[i for i,(x,y)in enumerate(zip(a,c))if x!=y],output_files=files,qualification=False,limitations=['Both runs strictly reject, zero accepted steps. Conditioning may change the Newton correction after a newly admitted restart vector; physical parity is required only before that change. Admission validation is separate.'])
assert report['identical_before_first_attempt']
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'audit.py').write_bytes(Path(__file__).read_bytes()); print(json.dumps(report,indent=2))
