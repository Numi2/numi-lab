import io, urllib.request, zipfile
URL='https://zenodo.org/records/7947104/files/formatted_baglione_crows.zip?download=1'
class Remote(io.RawIOBase):
 def __init__(self):
  req=urllib.request.Request(URL,method='HEAD',headers={'User-Agent':'NumiResearch/1.0'})
  with urllib.request.urlopen(req,timeout=30) as r:self.size=int(r.headers['Content-Length'])
  self.pos=0;self.blocks={};self.blocksize=1<<20
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,off,whence=0):
  self.pos=off if whence==0 else self.pos+off if whence==1 else self.size+off
  if self.pos<0:raise ValueError('negative seek')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos);out=bytearray()
  while n>0:
   block=self.pos//self.blocksize;start=block*self.blocksize;end=min(start+self.blocksize,self.size)-1
   if block not in self.blocks:
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'NumiResearch/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:
     if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('server did not honor requested range')
     self.blocks[block]=r.read()
    if len(self.blocks[block])!=end-start+1:raise RuntimeError('truncated range')
   chunk=self.blocks[block][self.pos-start:self.pos-start+n];out.extend(chunk);self.pos+=len(chunk);n-=len(chunk)
  return bytes(out)

import collections,gzip,hashlib,json,numpy as np,yaml
from pathlib import Path
A=Path(__file__).resolve().parent;r=Remote();meta={}
with zipfile.ZipFile(r) as z:
 prefix='formatted_baglione_crows/'
 metadata=z.read(prefix+'dataset_metadata.yaml');license=z.read(prefix+'license_bebe.txt');(A/'dataset_metadata.yaml').write_bytes(metadata);(A/'LICENSE-BEBE.txt').write_bytes(license);m=yaml.safe_load(metadata);assert m['sr']==50 and m['label_names']==['unknown','flying','in_nest']
 index=[];stats=[];all_lines=[]
 for bird in range(11):
  names=[x for x,k in m['clip_id_to_individual_id'].items() if k==bird];options=[z.getinfo(prefix+'clip_data/'+name+'.csv') for name in names];chosen=min(options,key=lambda x:x.file_size);chosen_id=chosen.filename.rsplit('/',1)[-1][:-4];selected=0;scanned=0;windows=[];last=-4
  with z.open(chosen) as f:
   for block in range(8640):
    rows=[f.readline() for _ in range(250)];nonempty=[line for line in rows if line];scanned+=len(nonempty)
    if len(nonempty)<250:break
    raw=b''.join(nonempty[:250]);a=np.loadtxt(io.BytesIO(raw),delimiter=',');assert a.shape==(250,5) and np.isfinite(a).all() and np.all(a[:,3]==bird)
    if block-last>=4 and np.all(a[:,4]==1):
     spectrum=np.fft.rfft((a[:,:3]-np.mean(a[:,:3],axis=0))*np.hanning(250)[:,None],axis=0);power=np.sum(np.abs(spectrum)**2,axis=1);freq=np.fft.rfftfreq(250,d=.02);keep=(freq>=1)&(freq<=12);ii=np.flatnonzero(keep)[np.argmax(power[keep])];peak=float(freq[ii]);concentration=float(power[(np.abs(freq-peak)<=.40000001)&keep].sum()/power[keep].sum()) if power[keep].sum()>0 else None
     info={'individual_id':bird,'split':'fit' if bird<8 else 'held_out','clip_id':chosen_id,'start_row':block*250,'row_count':250,'source_rows_sha256':hashlib.sha256(raw).hexdigest(),'peak_hz':peak,'band_concentration':concentration};index.append(info);windows.append(peak);all_lines.extend([f'# individual_id={bird} clip={chosen_id} start_row={block*250}\n'.encode(),raw]);selected+=1;last=block
    if selected>=12 or len(nonempty)<250:break
  stats.append({'individual_id':bird,'split':'fit' if bird<8 else 'held_out','clip_id':chosen_id,'zip_entry_crc32':f'{chosen.CRC:08x}','zip_entry_uncompressed_bytes':chosen.file_size,'rows_scanned':scanned,'windows':selected,'peak_median_hz':float(np.median(windows)) if windows else None});print('individual',bird,'windows',selected,'scanned',scanned,flush=True)
 data=b''.join(all_lines)
 with (A/'selected-flight-windows.csv.gz').open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz:gz.write(data)
 source={'record':'https://doi.org/10.5281/zenodo.7947104','zip_name':'formatted_baglione_crows.zip','source_zip_size':r.size,'published_md5_not_independently_verified':'a7b17e3beca03cdde3c11f0e310191fd','partial_entry_note':'Only initial bytes read; full ZIP/entry checksums not independently verified. HTTP range responses were checked and selected exact source rows hashed.','metadata_sha256':hashlib.sha256(metadata).hexdigest(),'license_sha256':hashlib.sha256(license).hexdigest(),'selected_uncompressed_sha256':hashlib.sha256(data).hexdigest(),'sample_rate_hz':50,'rows':index,'individuals':stats}
 (A/'source-sample-index.json').write_text(json.dumps(source,indent=2)+'\n')
 print('windows',len(index),'source_range_bytes',sum(map(len,r.blocks.values())),flush=True)
