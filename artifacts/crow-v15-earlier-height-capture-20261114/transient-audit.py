from pathlib import Path
import gzip,hashlib,json,statistics
A=Path('artifacts/crow-v15-earlier-height-capture-20261114');P=A/'turn-20261126-rebound-earlier';refpath=Path('artifacts/crow-v15-earlier-height-reference-20261114/reference.json.gz');refs={(x['lane'],x['step']):x for x in json.load(gzip.open(refpath,'rt'))};states={};dup=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];dup.extend([k] if k in states else []);states[k]=x
rows=[]
for e in (6,7):
 window=range(450,1001);a=[(states[e,t]['q'][2],t) for t in window];b=[(refs[e,t]['q'][2],t) for t in window];peak=max(a);baseline=max(b);wins=[]
 for lo,hi,goal in ((1000,1150,.55 if e==6 else .80),(1550,1800,.80 if e==6 else .55)):
  vals=[states[e,t]['q'][2] for t in range(lo,hi+1)];wins.append({'steps':[lo,hi],'target_m':goal,'mae_m':statistics.mean(abs(v-goal) for v in vals)})
 rows.append({'lane':e,'peak_height_m_and_step':peak,'baseline_peak_height_m_and_step':baseline,'peak_reduction_m':baseline[0]-peak[0],'peak_gate_pass':baseline[0]-peak[0]>=.10,'height_windows':wins,'height_gate_pass':all(x['mae_m']<=.05 for x in wins)})
out={'trace_sha256':hashlib.sha256((P/'run.stderr.gz').read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),'duplicate_states':dup,'rows':rows,'scope':'matched used-seed obstacle-free development trial'};(A/'transient-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
