from pathlib import Path
import gzip,json,sys,hashlib
source=Path(sys.argv[1]);out=Path(sys.argv[2]);data={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'state':[],'target':[],'contact':[],'leg':[],'toe':[],'aero':[],'errors':[]}
with gzip.open(source,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if x['step']<=500:data['state'].append(x)
   continue
  z=line.split()
  if not z:continue
  typ=z[0]
  if typ=='CROW_SUBSTEP_ERROR' or 'OBSERVER_ERROR' in typ:data['errors'].append(line.strip());continue
  if typ=='CROW_DRIVE_TARGET':
   t,e=map(int,z[1:3]);
   if 335<=t<=500:data['target'].append([e,t,*map(float,z[3:])])
  elif typ in ('CROW_TWO_TOE_SUBSTEP','CROW_LEG_EFFORT','CROW_TOE_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);
   if 335<=t<=500:
    if typ=='CROW_TWO_TOE_SUBSTEP':
     if int(z[4])>int(z[5]):data['errors'].append('contact overflow '+str((e,t,k)))
     data['contact'].append([e,t,k,*map(float,z[6:11])])
    elif typ=='CROW_LEG_EFFORT':data['leg'].append([e,t,k,*map(float,z[7:])])
    else:data['toe'].append([e,t,k,*map(float,z[4:])])
  elif typ=='CROW_AERO_FORCE':
   t,k,e=map(int,z[1:4]);
   if 335<=t<=500:data['aero'].append([e,t,k,*map(float,z[4:])])
with out.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(json.dumps(data,separators=(',',':')).encode())
print({k:len(v) for k,v in data.items() if isinstance(v,list)})
