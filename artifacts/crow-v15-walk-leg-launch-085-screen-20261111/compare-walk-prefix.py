from pathlib import Path
import gzip,json
D=Path('artifacts/crow-v15-walk-leg-launch-085-screen-20261111');P=Path('artifacts/crow-v15-walk-strength085-confirm-20261111')
def read(p):
 state={};sub={}
 for z in sorted(p.glob('stderr-*.gz')):
  with gzip.open(z,'rt') as f:
   for l in f:
    if l.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
     if t<=355:state[e,t]=x
    elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
     a=l.split();t,k,e=map(int,a[1:4])
     if t<=355:sub[e,t,k]=a[4:]
 return state,sub
rows=[]
for seed in (20261126,20261127):
 a,ac=read(D/('trial-on-'+str(seed)));b,bc=read(P/('trial-'+str(seed)))
 assert len(a)==len(b)==8*355 and len(ac)==len(bc)==8*355*16
 for e in range(8):
  diff=next((t for t in range(1,356) if any(a[e,t][k]!=b[e,t][k] for k in ('q','v','policy_action')) or any(ac[e,t,k]!=bc[e,t,k] for k in range(16))),None)
  rows.append({'seed':seed,'lane':e,'first_state_action_or_contact_difference':diff})
assert all(x['first_state_action_or_contact_difference'] is None for x in rows),rows
(D/'walk-prefix-parity.json').write_text(json.dumps(rows,indent=2)+'\n');print('exact matching lane/seed prefixes',len(rows))
