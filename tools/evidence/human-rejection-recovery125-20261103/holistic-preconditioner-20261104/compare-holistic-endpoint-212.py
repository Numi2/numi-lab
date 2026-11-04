from pathlib import Path
import hashlib,json,re,csv,gzip,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
def capture(p):
 f=p/'install-motion.stdout.log.gz'
 op=gzip.open if f.exists() else open
 if not f.exists():f=p/'install-motion.stdout.log'
 h=hashlib.sha256();counts={'node':0,'tet':0};begin=end=0;profiles=[];completed=[]
 with op(f,'rb') as src:
  for line in src:
   if line.startswith(b'resting_FEM_endpoint_capture=begin '):begin+=1
   if line.startswith(b'resting_FEM_endpoint_capture=end'):end+=1
   if line.startswith(b'resting_FEM_node,'):counts['node']+=1;h.update(line.strip()+b'\n')
   elif line.startswith(b'resting_FEM_tet,'):counts['tet']+=1;h.update(line.strip()+b'\n')
   elif line.startswith(b'resting_native_profile '):profiles.append(line.decode().strip())
   elif line.startswith(b'resting_integrated_body='):completed.append(line.decode().strip())
 assert counts=={'node':318974,'tet':1479479} and begin==end==1,(counts,begin,end)
 return {'fine_records_sha256':h.hexdigest(),'fine_record_counts':counts,'capture_complete':True,'profile':profiles,'completed':completed}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
a,b=D/'shared-three-rhs-native-190',D/'holistic-integrated-parity-native-212'
r={'baseline':str(a),'candidate':str(b),'baseline_capture':capture(a),'candidate_capture':capture(b)}
r['all_fine_records_equal']=r['baseline_capture']['fine_records_sha256']==r['candidate_capture']['fine_records_sha256']
r['files']=[]
for pattern,count in [('*.csv',11),('*.mrvpack',3)]:
 files=sorted((a/'install-motion').rglob(pattern));assert len(files)==count
 for x in files:
  rel=x.relative_to(a/'install-motion');y=b/'install-motion'/rel
  hx,hy=sha(x),sha(y)
  r['files'].append({'path':str(rel),'source_sha256':hx,'candidate_sha256':hy,'equal':hx==hy})
r['all_11_traces_and_3_captures_equal']=all(x['equal'] for x in r['files'])
r['qualification']='One full-coordinate15.625us release-transient endpoint. Exact fine records/traces/geometry are separate from initialization, sustained performance and supported-rest qualification. No physical force channel omitted by the existing owner endpoint capture is claimed covered.'
(D/'comparison-190-212.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
print('different_files',[x['path'] for x in r['files'] if not x['equal']])
sys.exit(0 if r['all_fine_records_equal'] and r['all_11_traces_and_3_captures_equal'] else 2)

