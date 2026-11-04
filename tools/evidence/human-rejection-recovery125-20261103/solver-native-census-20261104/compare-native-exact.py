from pathlib import Path
import sys,json,hashlib,gzip,re
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def output(p):
 f=p/'install-motion.stdout.log'
 return f.open('rb') if f.exists() else gzip.open(str(f)+'.gz','rb')
def endpoint(p):
 h=hashlib.sha256();n=0;summary=[];times=[];kernels=[]
 with output(p) as f:
  for b in f:
   if b'resting_FEM_' in b:
    at=b.find(b'resting_FEM_')
    if b[at:].startswith((b'resting_FEM_node,',b'resting_FEM_tet,',b'resting_FEM_endpoint_capture=')):
     h.update(b[at:]);n+=1
   if b.startswith(b'resting_native_profile '):
    summary.append(b.decode().strip())
    m=re.search(rb'gpu_ms=([0-9.eE+-]+)',b)
    if m:times.append(float(m[1]))
   if b.startswith(b'matter_gpu_timing=') or b.startswith(b'matter_stage_gpu_timing'):
    kernels.append(b.decode().strip())
 return {'sha256':h.hexdigest(),'lines':n,'profiles':summary,'gpu_ms':times}
a,b=D/sys.argv[1],D/sys.argv[2]
report={'baseline':str(a),'candidate':str(b)}
report['endpoint_reference']=endpoint(a);report['endpoint_candidate']=endpoint(b)
report['full_fine_endpoint_records_equal']=report['endpoint_reference']['lines']==1798455 and report['endpoint_reference']['sha256']==report['endpoint_candidate']['sha256']
for key,pat,expected in [('CSV','*.csv',11),('MRV','*.mrvpack',3)]:
 files=sorted((a/'install-motion').rglob(pat))
 assert len(files)==expected,(key,len(files))
 report[key]=[]
 for x in files:
  rel=x.relative_to(a/'install-motion');y=b/'install-motion'/rel
  hx,hy=sha(x),sha(y)
  report[key].append({'path':str(rel),'equal':hx==hy,'source_sha256':hx,'candidate_sha256':hy})
report['equivalent_algorithm_parity']=report['full_fine_endpoint_records_equal'] and all(x['equal'] for k in ['CSV','MRV'] for x in report[k])
report['qualification']='One-step full-coordinate scheduling/kernel parity; not supported rest, breathing motion, sustained throughput or0.1x.'
out=D/sys.argv[3];out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['CSV','MRV']},indent=2),flush=True)
sys.exit(0 if report['equivalent_algorithm_parity'] else 2)
