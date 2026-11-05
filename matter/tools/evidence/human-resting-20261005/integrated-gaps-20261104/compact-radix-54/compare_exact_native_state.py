from pathlib import Path
import argparse,hashlib,json,gzip,re,sys
a=argparse.ArgumentParser()
a.add_argument('reference',type=Path);a.add_argument('candidate',type=Path);a.add_argument('output',type=Path);v=a.parse_args()
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def endpoint(d):
 p=d/'install-motion.stdout.log.gz';h=hashlib.sha256();counts={'nodes':0,'tets':0};body=None
 with gzip.open(p,'rb')as f:
  for line in f:
   for prefix,key in [(b'resting_FEM_node,','nodes'),(b'resting_FEM_tet,','tets')]:
    at=line.find(prefix)
    if at>=0:h.update(line[at:].rstrip(b'\r\n')+b'\n');counts[key]+=1
   if b'resting_integrated_body=completed' in line:body=line.decode().strip()
 return {'path':str(p),'sha256':h.hexdigest(),'counts':counts,'body':body,'file_sha256':sha(p)}
ref=endpoint(v.reference);candidate=endpoint(v.candidate);csv={}
for p in sorted((v.reference/'install-motion').glob('*.csv')):
 q=v.candidate/'install-motion'/p.name
 row={'reference_sha256':sha(p),'candidate_sha256':sha(q),'bitwise_equal':sha(p)==sha(q)}
 if p.name=='resting-surface-audit.csv':
  # Optional native export emits a duplicate audit row; never erase differing
  # values. Retain strict-byte result and independently compare every unique row.
  row['unique_rows_equal']=set(p.read_bytes().splitlines())==set(q.read_bytes().splitlines())
 csv[p.name]=row
rows=[];timings=[]
for line in (v.candidate/'install-motion.stderr.log').read_text().splitlines():
 if line.startswith('matter_fgmres_active_work='):rows.append(json.loads(line.split('=',1)[1]))
 if line.startswith('human_preparation_memory phase=accepted_step_'):timings.append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
out={'reference_directory':str(v.reference),'candidate_directory':str(v.candidate),'reference_endpoint':ref,'candidate_endpoint':candidate,
 'endpoint_bitwise_equal':ref['sha256']==candidate['sha256'] and ref['counts']==candidate['counts'],
 'compact_csv':csv,'roots_recorded':len(rows),'active_work':{k:sum(x[k]for x in rows)for k in ['cycles_with_work','used_columns_sum','budget_exhausted_solves']},'accepted_memory':timings[-1:] ,
 'scope':'32-root release transient / 0.5 ms only; not supported rest, physiological plausibility or sustained end-to-end qualification'}
out['pass_exact_bytes']=out['endpoint_bitwise_equal'] and all(x['bitwise_equal']for x in csv.values())
out['pass_numerical_state']=out['endpoint_bitwise_equal'] and candidate['counts']=={'nodes':318974,'tets':1479479} and len(csv)==10 and len(rows)==32 and all(x['bitwise_equal']or(k=='resting-surface-audit.csv'and x['unique_rows_equal'])for k,x in csv.items())
v.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k!='compact_csv'}))
sys.exit(0 if out['pass_numerical_state'] else 1)
