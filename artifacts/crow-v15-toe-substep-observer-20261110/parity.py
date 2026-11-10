from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-toe-substep-observer-20261110');name=sys.argv[1];reference=Path(sys.argv[2])
with gzip.open(reference,'rt') as f:a=json.load(f)['payload']['frames']
with gzip.open(A/(name+'.json.gz'),'rt') as f:b=json.load(f)['payload']['frames']
keys=('step','q','v','actor_observation','accepted_actions');diff=[]
for x,y in zip(a,b):
 for k in keys:
  if x[k]!=y[k] and len(diff)<10:diff.append({'step':x['step'],'field':k})
r={'reference':str(reference),'reference_frames':len(a),'observed_frames':len(b),'equal_all_selected_fields':len(a)==len(b) and not diff,'first_differences':diff,'scope':list(keys)};(A/(name+'-parity.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));sys.exit(0 if r['equal_all_selected_fields'] else 1)
