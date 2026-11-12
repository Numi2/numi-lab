from pathlib import Path
import gzip,hashlib,io,json
class Joined(io.RawIOBase):
 def __init__(self,parts):
  self.parts=iter(parts);self.current=None
 def readable(self):return True
 def readinto(self,buf):
  while True:
   if self.current is None:
    try:self.current=next(self.parts).open('rb')
    except StopIteration:return 0
   n=self.current.readinto(buf)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current:self.current.close()
  super().close()
def verified_parts(directory):
 directory=Path(directory);m=json.loads((directory/'stderr-parts.json').read_text());total=hashlib.sha256();size=0;parts=[]
 for x in m['parts']:
  p=directory/x['name'];h=hashlib.sha256();n=0
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b);total.update(b);n+=len(b)
  if h.hexdigest()!=x['sha256'] or n!=x['bytes']:raise ValueError('corrupt part '+str(p))
  parts.append(p);size+=n
 if total.hexdigest()!=m['compressed_sha256'] or size!=m['bytes']:raise ValueError('combined hash mismatch')
 return parts
def open_text(path):
 p=Path(path)
 if (p.parent/'stderr-parts.json').exists():
  return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BufferedReader(Joined(verified_parts(p.parent))),mode='rb'))
 return gzip.open(p,'rt')
if __name__=='__main__':
 import sys
 count=0;digest=hashlib.sha256()
 with open_text(Path(sys.argv[1])/'run.stderr.gz') as f:
  for line in f:count+=1;digest.update(line.encode())
 print(json.dumps({'lines':count,'uncompressed_sha256':digest.hexdigest()}))
