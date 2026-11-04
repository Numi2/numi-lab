from pathlib import Path
import hashlib,json,re,sys,gzip
w=Path(__file__).resolve().parent
d=w/'specialized-cooperative-tangents-284'
ref=w/'compact-immutable-contacts-276'
assert (d/'install-motion.execution.json').exists()
def filehash(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def records(p):
 h=hashlib.sha256();counts={'nodes':0,'tets':0};body=None
 with (gzip.open(p,'rb') if p.suffix=='.gz' else p.open('rb')) as f:
  for line in f:
   for prefix,key in [(b'resting_FEM_node,','nodes'),(b'resting_FEM_tet,','tets')]:
    n=line.find(prefix)
    if n>=0:
     h.update(line[n:].rstrip(b'\r\n')+b'\n');counts[key]+=1
   if b'resting_integrated_body=completed' in line:body=line.decode().strip()
 return {'sha256':h.hexdigest(),'counts':counts,'body':body,'file_sha256':filehash(p)}
candidate=records(d/'install-motion.stdout.log.gz')
baseline=records(ref/'install-motion.stdout.log.gz')
csv={}
for p in sorted((ref/'install-motion').glob('*.csv')):
 q=d/'install-motion'/p.name
 csv[p.name]={'reference_sha256':filehash(p),'candidate_sha256':filehash(q)}
 csv[p.name]['bitwise_equal']=csv[p.name]['reference_sha256']==csv[p.name]['candidate_sha256']
active=[];storage=[];memory=[]
text=(d/'install-motion.stderr.log').read_text()
for line in text.splitlines():
 if 'matter_fgmres_active_work=' in line:active.append(json.loads(line.split('=',1)[1]))
 if 'human_compact_immutable_contacts ' in line:storage.append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
 if 'human_preparation_memory phase=' in line:memory.append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
out={'reference_directory':str(ref),'candidate_directory':str(d),'endpoint_reference':baseline,'endpoint_candidate':candidate,'endpoint_bitwise_equal':candidate['sha256']==baseline['sha256'],'compact_csv':csv,'root_count':len(active),'active_work':{k:sum(x[k]for x in active)for k in ['cycles_with_work','used_columns_sum','budget_exhausted_solves']},'storage':storage,'memory':memory,'scope':'32-root release transient, 0.5 ms; no settled-rest or sustained physiology qualification'}
out['pass_exact_state']=out['endpoint_bitwise_equal'] and candidate['counts']=={'nodes':318974,'tets':1479479} and len(csv)==10 and all(x['bitwise_equal']for x in csv.values()) and out['root_count']==32
(w/'specialized-cooperative-284-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['compact_csv','memory']}))
if not out['pass_exact_state']:sys.exit(1)
