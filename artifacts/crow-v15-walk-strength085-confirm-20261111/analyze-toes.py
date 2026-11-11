from pathlib import Path
import gzip,json,sys
seed=int(sys.argv[1]);assert seed in (20261126,20261127);D=Path('artifacts/crow-v15-walk-strength085-confirm-20261111');hits={e:[] for e in range(8)};seen=set();maxspeed={e:0. for e in range(8)}
for p in sorted((D/('trial-'+str(seed))).glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_TOE_V_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));assert len(v)==16
    if not 1<=t<=355:continue
    assert (e,t,k) not in seen;seen.add((e,t,k));maxspeed[e]=max(maxspeed[e],max(map(abs,v)))
    if any(abs(z)>=11.999 for z in v):hits[e].append([t,k,[i for i,z in enumerate(v) if abs(z)>=11.999]])
assert len(seen)==8*355*16,len(seen)
r={'seed':seed,'lanes':[{'lane':e,'max_abs_toe_speed_rad_s':maxspeed[e],'all_hits':hits[e],'gait_hits':[h for h in hits[e] if 201<=h[0]<=301]} for e in range(8)],'scope':'All 16 toe velocities at all 16 substeps over 355 first-episode intervals. A velocity below the configured cap does not establish biological realism.'}
(D/('toe-analysis-'+str(seed)+'.json')).write_text(json.dumps(r,indent=2)+'\n');print([(x['lane'],len(x['all_hits']),len(x['gait_hits'])) for x in r['lanes']])
