from pathlib import Path
import gzip,math,json,hashlib
A=Path('artifacts/crow-v15-flight-speed-gain-screen-20261113');rows=[];valid=True
for arm in ('push',):
 p=A/(arm+'-wing-20261126-rebound-gain')/'run.stderr.gz';flags=[];old={};new={};scales={}
 with gzip.open(p,'rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_DRIVE_FLAGS':flags.append(list(map(int,z[1:])))
   elif z[0]=='CROW_MODEL_DRIVE':old[int(z[1])]=list(map(float,z[2:]))
   elif z[0]=='CROW_MODEL_DRIVE_NEW':new[int(z[1])]=list(map(float,z[2:]))
   elif z[0]=='CROW_DRIVE_SCALES':
    t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in scales;scales[key]=list(map(float,z[4:]));assert len(scales[key])==4
 assert len(flags)==1 and flags[0][0]==1 and flags[0][1:3]==[0,0] and flags[0][3]&32 and flags[0][4]&4 and flags[0][5]&1,flags
 assert set(old)==set(new)=={8,9} and all(old[i][0:4]==[i+1,i,800.,8.] and old[i][4]>0 and old[i][5]==120 for i in (8,9)),old
 assert all(new[i][0]==10 and abs(new[i][1]-.65)<1e-7 and new[i][2:]==old[i][4:] for i in (8,9)),new
 assert len(scales)==8*153*16 and all(all(math.isfinite(v) for v in x) for x in scales.values())
 unique=sorted({tuple(x) for x in scales.values()});is_unit=all(abs(x[0]-1)<1e-7 and abs(x[1]-1)<1e-7 and abs(x[3]-1)<1e-7 for x in scales.values());valid=valid and is_unit
 rows.append({'arm':arm,'flags':flags[0],'old_dofs':old,'new_dofs':new,'distinct_scales':unique,'unit_gain_scales':is_unit,'log_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(A/'drive-validation.json').write_text(json.dumps({'valid':valid,'rows':rows,'scope':'Checks uploaded model gains, host dispatch flags, and sampled controller scales. It does not independently inspect every GPU internal value or qualify dynamics.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
if not valid:raise SystemExit('gain scale differs; consistency claim not established')
