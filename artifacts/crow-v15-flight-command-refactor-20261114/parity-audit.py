from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-flight-command-refactor-20261114');P=A/sys.argv[1];ref={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'reference-states.json.gz','rt'))};seen=set();bad=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);assert key not in seen;seen.add(key);y=ref.get(key)
  if y is None or any(x[k]!=y[k] for k in ('q','v','policy_action','contacts')):bad.append(key)
missing=sorted(set(ref)-seen);extra=sorted(seen-set(ref));result={'reference_records':len(ref),'candidate_records':len(seen),'mismatch_count':len(bad),'first_mismatches':bad[:20],'missing':missing[:20],'missing_count':len(missing),'extra':extra[:20],'extra_count':len(extra),'passed':len(ref)==8*1801 and len(seen)==len(ref) and not bad and not missing and not extra,'scope':'exact logged q/v/policy-action/contact comparison; does not prove arbitrary command response'};(P/'full-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
