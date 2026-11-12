from pathlib import Path
import gzip,json,subprocess
A=Path('artifacts/crow-v15-terminal-descent-damping-screen-20261112');B=Path('artifacts/crow-v15-localized-route-autonomous-eval-20261111/autonomous-20261126');C=A/'autonomous-20261126-screen'
def stream(d):
 p=subprocess.Popen(['cat',*[str(x) for x in sorted(d.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE)
 with gzip.open(p.stdout,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):yield json.loads(line.split(' ',1)[1])
 assert p.wait()==0
first={};counts={i:0 for i in range(8)}
for a,b in zip(stream(B),stream(C)):
 assert (a['lane'],a['step'])==(b['lane'],b['step'])
 e=a['lane'];counts[e]+=1
 if a!=b and e not in first:first[e]=a['step']
r={'first_state_divergence':first,'compared_endpoints':counts,'scope':'Control endpoint record, including actor outputs; this is not a full-substep verification before each divergence.'};(A/'first-divergence.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
